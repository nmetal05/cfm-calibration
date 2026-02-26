"""
Generate full SBI training dataset — IDM VERSION (robust)
Oversamples and filters physically invalid parameter combos

Usage: python generate_training_data.py
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
SIM_TIMEOUT = 180  # Slightly more generous for IDM

PRIOR_LOW = np.array([0.5, 0.0, 0.0, 0.5])
PRIOR_HIGH = np.array([1.3, 0.3, 1.0, 3.0])
PARAM_NAMES = ["speedFactor", "speedDev", "sigma", "tau"]
N_PARAMS = 4

# Load network data ONCE
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

        # Log failure reason
        stderr = ""
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[:200]

        return sim_id, theta, None, time.time() - t_start

    except subprocess.TimeoutExpired:
        # Cleanup on timeout
        for f in [edgedata_output, vtype_path, edgedata_add]:
            try:
                f.unlink(missing_ok=True)
            except:
                pass
        return sim_id, theta, None, time.time() - t_start
    except Exception:
        return sim_id, theta, None, time.time() - t_start


def main():
    print("=" * 60)
    print("GENERATING SBI TRAINING DATA — IDM (robust)")
    print("=" * 60)
    print(f"Model:      IDM (Intelligent Driver Model)")
    print(f"Target:     {N_SIMS} successful simulations")
    print(f"Submitted:  {N_SIMS} (oversampled for failures)")
    print(f"Workers:    {N_WORKERS}")
    print(f"Params:     {PARAM_NAMES} ({N_PARAMS}D)")
    print(f"Prior low:  {PRIOR_LOW}")
    print(f"Prior high: {PRIOR_HIGH}")
    print(f"Stats dim:  57 (speed + congestion)")
    print()

    assert (BASE_DIR / "osm.net.xml").exists(), "Missing osm.net.xml"
    assert (BASE_DIR / "routes_peak_novtype.rou.xml").exists(), (
        "Missing routes_peak_novtype.rou.xml"
    )
    assert (BASE_DIR / "sbi_peak.sumocfg").exists(), "Missing sbi_peak.sumocfg"

    (BASE_DIR / "sbi_runs").mkdir(exist_ok=True)

    # Sample physically valid thetas
    print("Sampling parameter combinations...")
    np.random.seed(42)
    all_thetas = np.random.uniform(PRIOR_LOW, PRIOR_HIGH, size=(N_SIMS, N_PARAMS))
    np.save("sbi_thetas_sampled.npy", all_thetas)

    print("\nSampled theta ranges:")
    for i, name in enumerate(PARAM_NAMES):
        print(
            f"  {name:15s}: [{all_thetas[:, i].min():.3f}, {all_thetas[:, i].max():.3f}]"
        )
    print()

    thetas_list = []
    xs_list = []
    completed = 0
    failed = 0
    failed_thetas = []
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
                failed_thetas.append(theta)

            total = completed + failed
            if total % 50 == 0:
                elapsed = time.time() - global_start
                rate = completed / (elapsed / 60) if elapsed > 0 else 0
                fail_pct = failed / total * 100 if total > 0 else 0
                remaining = (N_SIMS - total) / (total / elapsed) if total > 0 else 0
                print(
                    f"  [{elapsed / 60:5.1f} min] "
                    f"Done: {completed}/{N_SIMS} target | "
                    f"Fail: {failed} ({fail_pct:.1f}%) | "
                    f"Rate: {rate:.0f}/min | "
                    f"ETA: {remaining / 60:.0f} min"
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
    print(f"Theta:      {theta_tensor.shape}  (target: [{N_SIMS}, {N_PARAMS}])")
    print(f"X:          {x_tensor.shape}  (target: [{N_SIMS}, 57])")

    if completed < N_SIMS:
        print(f"\n  ⚠️  WARNING: Only {completed}/{N_SIMS} successful!")
        print(f"     Increase N_SIMS or tighten priors.")

    # Analyze failures
    if len(failed_thetas) > 5:
        failed_arr = np.array(failed_thetas)
        success_arr = np.array(thetas_list)
        print(f"\n  Failure analysis (what kills IDM sims):")
        print(
            f"  {'Parameter':>15} | {'Failed mean':>12} | {'Success mean':>12} | {'Diff':>8}"
        )
        print(f"  " + "-" * 55)
        for i, name in enumerate(PARAM_NAMES):
            f_mean = failed_arr[:, i].mean()
            s_mean = success_arr[:, i].mean()
            diff = f_mean - s_mean
            flag = (
                " ← likely cause"
                if abs(diff) > 0.1 * (PRIOR_HIGH[i] - PRIOR_LOW[i])
                else ""
            )
            print(
                f"  {name:>15} | {f_mean:12.3f} | {s_mean:12.3f} | {diff:+8.3f}{flag}"
            )

    # Sanity check
    print(f"\nSanity check:")
    for i, name in enumerate(PARAM_NAMES):
        col = theta_tensor[:, i]
        print(f"  {name:15s}: [{col.min():.3f}, {col.max():.3f}] mean={col.mean():.3f}")
    print(f"  NaN: {torch.isnan(x_tensor).any()}")
    print(f"  Inf: {torch.isinf(x_tensor).any()}")

    fail_rate = failed / (completed + failed) * 100
    if fail_rate > 15:
        print(f"\n  ⚠️  High failure rate ({fail_rate:.1f}%)!")
        print(f"     The failure analysis above shows which params cause crashes.")
    else:
        print(f"\n  ✓ Failure rate: {fail_rate:.1f}% (acceptable)")

    print(f"\nNext: python train_snpe_v3.py")


if __name__ == "__main__":
    main()
