"""
BENCHMARK: How many SUMO simulations can we run in 20 minutes?
Usage: python benchmark.py
"""

import os
import time
import subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from write_vtype import write_vtype_file
from parse_output import parse_edge_data, summary_statistics

# =============================================================
SUMO_BINARY = "sumo"
BASE_DIR = Path(".")
N_WORKERS = 15
TIME_LIMIT_SECONDS = 20 * 60
MAX_SIMS = 2000
SIM_TIMEOUT = 120          # 2 min timeout (sims take ~10s now)
# =============================================================

PRIOR_LOW = np.array([0.7, 0.0, 0.0, 0.5])
PRIOR_HIGH = np.array([1.2, 0.25, 1.0, 3.0])


def run_single_sim(sim_id):
    """Run one SUMO simulation with random parameters"""
    sim_dir = BASE_DIR / "sbi_runs" / f"sim_{sim_id:05d}"
    sim_dir.mkdir(parents=True, exist_ok=True)

    theta = np.random.uniform(PRIOR_LOW, PRIOR_HIGH)

    # Write vType
    vtype_path = sim_dir / "vtype.xml"
    write_vtype_file(theta, str(vtype_path))

    # Write edgeData config
    edgedata_output = sim_dir / "edgedata.xml"
    edgedata_add = sim_dir / "edgedata.add.xml"
    with open(edgedata_add, "w", encoding="utf-8") as f:
        f.write(
            '<additional>\n'
            f'    <edgeData id="sbi" freq="300" file="{edgedata_output.resolve()}" excludeEmpty="true"/>\n'
            '</additional>\n'
        )

    # Delete old output if exists
    if edgedata_output.exists():
        edgedata_output.unlink()

    t_start = time.time()
    try:
        result = subprocess.run(
            [
                SUMO_BINARY,
                "-c", str((BASE_DIR / "sbi_peak.sumocfg").resolve()),
                "--additional-files", f"{vtype_path.resolve()},{edgedata_add.resolve()}",
                "--seed", str(sim_id),
            ],
            capture_output=True,
            timeout=SIM_TIMEOUT,
            cwd=str(BASE_DIR.resolve())
        )
        elapsed = time.time() - t_start

        if result.returncode == 0 and edgedata_output.exists():
            edge_data = parse_edge_data(str(edgedata_output))
            x = summary_statistics(edge_data)

            # Cleanup to save disk
            edgedata_output.unlink(missing_ok=True)

            return sim_id, theta, x, elapsed
        else:
            return sim_id, theta, None, elapsed

    except subprocess.TimeoutExpired:
        return sim_id, theta, None, time.time() - t_start
    except Exception as e:
        return sim_id, theta, None, time.time() - t_start


def main():
    print("=" * 60)
    print("SUMO SBI BENCHMARK")
    print("=" * 60)
    print(f"Workers:        {N_WORKERS}")
    print(f"Time limit:     {TIME_LIMIT_SECONDS // 60} minutes")
    print(f"Sim timeout:    {SIM_TIMEOUT}s")
    print(f"Prior low:      {PRIOR_LOW}")
    print(f"Prior high:     {PRIOR_HIGH}")
    print()

    # Check prerequisites
    assert (BASE_DIR / "osm.net.xml").exists(), "osm.net.xml not found!"
    assert (BASE_DIR / "routes_peak_novtype.rou.xml").exists(), \
        "routes_peak_novtype.rou.xml not found! Run fix_routes.py first!"
    assert (BASE_DIR / "sbi_peak.sumocfg").exists(), "sbi_peak.sumocfg not found!"

    (BASE_DIR / "sbi_runs").mkdir(exist_ok=True)

    completed = 0
    failed = 0
    sim_times = []
    global_start = time.time()

    print(f"Starting at {time.strftime('%H:%M:%S')}")
    print("-" * 60)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {}
        for i in range(MAX_SIMS):
            future = executor.submit(run_single_sim, i)
            futures[future] = i

        for future in as_completed(futures):
            elapsed_total = time.time() - global_start

            if elapsed_total > TIME_LIMIT_SECONDS:
                print(f"\nTime limit reached ({TIME_LIMIT_SECONDS // 60} min)")
                for f in futures:
                    f.cancel()
                break

            result = future.result()
            if result is None:
                failed += 1
                continue

            sim_id, theta, x, sim_time = result

            if x is not None:
                completed += 1
                sim_times.append(sim_time)
            else:
                failed += 1

            if (completed + failed) % 10 == 0:
                rate = completed / (elapsed_total / 60) if elapsed_total > 0 else 0
                avg_time = np.mean(sim_times) if sim_times else 0
                print(
                    f"  [{elapsed_total:6.1f}s] "
                    f"Done: {completed} | "
                    f"Fail: {failed} | "
                    f"Rate: {rate:.1f}/min | "
                    f"Avg: {avg_time:.1f}s/sim"
                )

    total_time = time.time() - global_start

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total time:        {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"Completed:         {completed}")
    print(f"Failed:            {failed}")
    print()

    if sim_times:
        sims_per_hour = completed / (total_time / 3600)
        print(f"Per sim:  mean={np.mean(sim_times):.1f}s  "
              f"median={np.median(sim_times):.1f}s  "
              f"min={np.min(sim_times):.1f}s  "
              f"max={np.max(sim_times):.1f}s")
        print()
        print(f"Throughput:        {sims_per_hour:.0f} sims/hour")
        print()
        print(f"SBI PROJECTIONS:")
        print(f"  3,000 sims:  ~{3000 / sims_per_hour:.1f} hours")
        print(f"  5,000 sims:  ~{5000 / sims_per_hour:.1f} hours")
        print(f"  10,000 sims: ~{10000 / sims_per_hour:.1f} hours")
    else:
        print("No simulations completed!")


if __name__ == "__main__":
    main()