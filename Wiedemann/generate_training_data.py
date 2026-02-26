"""
Generate full SBI training dataset — FINAL VERSION
Uses shared summary statistics (speed + congestion)

Usage: python generate_training_data.py
Expected: ~2 hours for 1000 simulations
"""

import os
import time
import subprocess
import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from write_vtype import write_vtype_file
from parse_output_shared import (
    parse_edge_data,
    summary_statistics_sumo,
    get_edge_max_speeds,
)

SUMO_BINARY = "sumo"
BASE_DIR = Path(".")
N_WORKERS = 15
N_SIMS = 1000
SIM_TIMEOUT = 120

PRIOR_LOW = np.array([0.5, 0.0, 0.0, 0.5])
PRIOR_HIGH = np.array([1.3, 0.3, 1.0, 3.0])
PARAM_NAMES = ["speedFactor", "speedDev", "sigma", "tau"]
N_PARAMS = 4

# Load network data ONCE (shared across all sims)
print("Loading network edge speeds (one time only)...")
EDGE_MAX_SPEEDS, EDGE_LENGTHS = get_edge_max_speeds("osm.net.xml")
print(f"  Loaded {len(EDGE_MAX_SPEEDS)} edge speeds")


def run_single_sim(args):
    """Run one SUMO simulation"""
    sim_id, theta = args

    sim_dir = BASE_DIR / "sbi_runs" / f"sim_{sim_id:05d}"
    sim_dir.mkdir(parents=True, exist_ok=True)

    vtype_path = sim_dir / "vtype.xml"
    write_vtype_file(theta.tolist(), str(vtype_path))

    edgedata_output = sim_dir / "edgedata.xml"
    edgedata_add = sim_dir / "edgedata.add.xml"
    with open(edgedata_add, "w", encoding="utf-8") as f:
        f.write(
            "<additional>\n"
            f'    <edgeData id="sbi" freq="300" '
            f'file="{edgedata_output.resolve()}" excludeEmpty="true"/>\n'
            "</additional>\n"
        )

    if edgedata_output.exists():
        edgedata_output.unlink()

    t_start = time.time()
    try:
        result = subprocess.run(
            [
                SUMO_BINARY,
                "-c",
                str((BASE_DIR / "sbi_peak.sumocfg").resolve()),
                "--additional-files",
                f"{vtype_path.resolve()},{edgedata_add.resolve()}",
                "--seed",
                str(sim_id),
            ],
            capture_output=True,
            timeout=SIM_TIMEOUT,
            cwd=str(BASE_DIR.resolve()),
        )
        elapsed = time.time() - t_start

        if result.returncode == 0 and edgedata_output.exists():
            edge_data = parse_edge_data(str(edgedata_output))
            x = summary_statistics_sumo(edge_data, EDGE_MAX_SPEEDS, EDGE_LENGTHS)

            # Cleanup
            edgedata_output.unlink(missing_ok=True)
            vtype_path.unlink(missing_ok=True)
            edgedata_add.unlink(missing_ok=True)
            try:
                sim_dir.rmdir()
            except OSError:
                pass

            if x is not None and len(x) == 57 and np.all(np.isfinite(x)):
                return sim_id, theta, x, elapsed

        return sim_id, theta, None, time.time() - t_start

    except subprocess.TimeoutExpired:
        return sim_id, theta, None, time.time() - t_start
    except Exception:
        return sim_id, theta, None, time.time() - t_start


def main():
    print("=" * 60)
    print("GENERATING SBI TRAINING DATA")
    print("=" * 60)
    print(f"Target:     {N_SIMS} simulations")
    print(f"Workers:    {N_WORKERS}")
    print(f"Params:     {PARAM_NAMES}")
    print(f"Prior low:  {PRIOR_LOW}")
    print(f"Prior high: {PRIOR_HIGH}")
    print(f"Stats dim:  57 (speed + congestion)")
    print(f"Est. time:  ~{N_SIMS / 2500:.1f} hours")
    print()

    assert (BASE_DIR / "osm.net.xml").exists()
    assert (BASE_DIR / "routes_peak_novtype.rou.xml").exists()
    assert (BASE_DIR / "sbi_peak.sumocfg").exists()

    (BASE_DIR / "sbi_runs").mkdir(exist_ok=True)

    # Sample thetas
    np.random.seed(42)
    all_thetas = np.random.uniform(PRIOR_LOW, PRIOR_HIGH, size=(N_SIMS, N_PARAMS))
    np.save("sbi_thetas_sampled.npy", all_thetas)

    thetas_list = []
    xs_list = []
    completed = 0
    failed = 0
    global_start = time.time()

    print(f"Starting at {time.strftime('%H:%M:%S')}")
    print("-" * 60)

    args = [(i, all_thetas[i]) for i in range(N_SIMS)]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(run_single_sim, arg): arg[0] for arg in args}

        for future in as_completed(futures):
            result = future.result()
            if result is None:
                failed += 1
                continue

            sim_id, theta, x, sim_time = result

            if x is not None:
                thetas_list.append(theta)
                xs_list.append(x)
                completed += 1
            else:
                failed += 1

            total = completed + failed
            if total % 50 == 0:
                elapsed = time.time() - global_start
                rate = completed / (elapsed / 60) if elapsed > 0 else 0
                remaining = (N_SIMS - total) / rate if rate > 0 else 0
                print(
                    f"  [{elapsed / 60:5.1f} min] "
                    f"Done: {completed}/{N_SIMS} | "
                    f"Fail: {failed} | "
                    f"Rate: {rate:.0f}/min | "
                    f"ETA: {remaining:.0f} min"
                )

            if completed > 0 and completed % 500 == 0:
                np.save("sbi_thetas_checkpoint.npy", np.array(thetas_list))
                np.save("sbi_xs_checkpoint.npy", np.array(xs_list))
                print(f"    [Checkpoint: {completed} sims saved]")

    total_time = time.time() - global_start

    # Save final
    theta_tensor = torch.tensor(np.array(thetas_list), dtype=torch.float32)
    x_tensor = torch.tensor(np.array(xs_list), dtype=torch.float32)

    torch.save(theta_tensor, "sbi_thetas.pt")
    torch.save(x_tensor, "sbi_xs.pt")

    print()
    print("=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Time:       {total_time / 60:.1f} min ({total_time / 3600:.1f} hours)")
    print(f"Completed:  {completed}")
    print(f"Failed:     {failed}")
    print(f"Theta:      {theta_tensor.shape}")
    print(f"X:          {x_tensor.shape}")

    # Sanity check
    print(f"\nSanity check:")
    for i, name in enumerate(PARAM_NAMES):
        col = theta_tensor[:, i]
        print(f"  {name:15s}: [{col.min():.3f}, {col.max():.3f}] mean={col.mean():.3f}")
    print(f"  NaN: {torch.isnan(x_tensor).any()}")
    print(f"  Inf: {torch.isinf(x_tensor).any()}")
    print(f"\nNext: python train_sbi.py")


if __name__ == "__main__":
    main()
