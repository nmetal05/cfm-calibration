"""
Run 10 simulations with different parameters, collect all data,
save to CSV so you can see exactly what's going on.

Usage: python inspect_data.py
"""

import time
import subprocess
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from write_vtype import write_vtype_file

SUMO_BINARY = "sumo"
BASE_DIR = Path(".")

PRIOR_LOW = np.array([0.7, 0.0, 0.0, 0.5])
PRIOR_HIGH = np.array([1.2, 0.25, 1.0, 3.0])
PARAM_NAMES = ["speedFactor", "speedDev", "sigma", "tau"]


def parse_edge_data_full(filepath):
    """Parse edgeData into a full DataFrame (not just summary stats)"""
    tree = ET.parse(filepath)
    root = tree.getroot()
    rows = []
    for interval in root.findall("interval"):
        t_begin = float(interval.get("begin"))
        t_end = float(interval.get("end"))
        for edge in interval.findall("edge"):
            rows.append({
                "time_begin": t_begin,
                "time_end": t_end,
                "time_hour": t_begin / 3600,
                "edge_id": edge.get("id"),
                "meanSpeed": float(edge.get("speed", 0)),
                "density": float(edge.get("density", 0)),
                "occupancy": float(edge.get("occupancy", 0)),
                "n_vehicles": int(edge.get("entered", 0)),
                "n_left": int(edge.get("left", 0)),
                "waitingTime": float(edge.get("waitingTime", 0)),
                "timeLoss": float(edge.get("timeLoss", 0)),
                "traveltime": float(edge.get("traveltime", 0)),
            })
    return pd.DataFrame(rows)


def run_one_sim(sim_id, theta):
    """Run one simulation, return raw DataFrame"""
    sim_dir = BASE_DIR / "inspect_runs" / f"sim_{sim_id:02d}"
    sim_dir.mkdir(parents=True, exist_ok=True)

    vtype_path = sim_dir / "vtype.xml"
    write_vtype_file(theta, str(vtype_path))

    edgedata_output = sim_dir / "edgedata.xml"
    edgedata_add = sim_dir / "edgedata.add.xml"
    with open(edgedata_add, "w", encoding="utf-8") as f:
        f.write(
            '<additional>\n'
            f'    <edgeData id="sbi" freq="300" '
            f'file="{edgedata_output.resolve()}" excludeEmpty="true"/>\n'
            '</additional>\n'
        )

    if edgedata_output.exists():
        edgedata_output.unlink()

    t_start = time.time()
    result = subprocess.run(
        [
            SUMO_BINARY,
            "-c", str((BASE_DIR / "sbi_peak.sumocfg").resolve()),
            "--additional-files", f"{vtype_path.resolve()},{edgedata_add.resolve()}",
            "--seed", str(sim_id),
        ],
        capture_output=True,
        timeout=120,
        cwd=str(BASE_DIR.resolve()),
    )
    elapsed = time.time() - t_start

    if result.returncode == 0 and edgedata_output.exists():
        df = parse_edge_data_full(str(edgedata_output))
        df["sim_id"] = sim_id
        for i, name in enumerate(PARAM_NAMES):
            df[name] = theta[i]
        return df, elapsed
    else:
        print(f"  Sim {sim_id} FAILED: {result.stderr[:200]}")
        return None, elapsed


def compute_summary_row(sim_id, theta, raw_df):
    """Compute summary statistics for one simulation - one row"""
    row = {"sim_id": sim_id}

    # Add theta values
    for i, name in enumerate(PARAM_NAMES):
        row[f"theta_{name}"] = theta[i]

    # Per time-bin mean speeds
    time_bins = sorted(raw_df["time_begin"].unique())
    for t in time_bins:
        bin_data = raw_df[raw_df["time_begin"] == t]
        hour_min = f"{int(t//3600)}:{int((t%3600)//60):02d}"
        row[f"meanSpeed_{hour_min}"] = bin_data["meanSpeed"].mean()
        row[f"stdSpeed_{hour_min}"] = bin_data["meanSpeed"].std()
        row[f"nVehicles_{hour_min}"] = bin_data["n_vehicles"].sum()

    # Global statistics
    row["global_meanSpeed"] = raw_df["meanSpeed"].mean()
    row["global_stdSpeed"] = raw_df["meanSpeed"].std()
    row["global_medianSpeed"] = raw_df["meanSpeed"].median()
    row["global_p10_speed"] = raw_df["meanSpeed"].quantile(0.10)
    row["global_p25_speed"] = raw_df["meanSpeed"].quantile(0.25)
    row["global_p75_speed"] = raw_df["meanSpeed"].quantile(0.75)
    row["global_p90_speed"] = raw_df["meanSpeed"].quantile(0.90)
    row["frac_below_5ms"] = (raw_df["meanSpeed"] < 5.0).mean()
    row["frac_below_3ms"] = (raw_df["meanSpeed"] < 3.0).mean()
    row["total_vehicles"] = raw_df["n_vehicles"].sum()
    row["total_waitingTime"] = raw_df["waitingTime"].sum()
    row["total_timeLoss"] = raw_df["timeLoss"].sum()
    row["n_edges"] = raw_df["edge_id"].nunique()
    row["n_time_bins"] = len(time_bins)

    return row


def main():
    print("=" * 60)
    print("INSPECT SIMULATION DATA")
    print("=" * 60)
    print("Running 10 simulations with varied parameters...")
    print()

    (BASE_DIR / "inspect_runs").mkdir(exist_ok=True)

    # Generate 10 different theta values
    # Include extremes and defaults to see the range
    thetas = [
        [0.9, 0.10, 0.35, 1.5],   # 0: default type_3
        [0.7, 0.00, 0.00, 0.5],   # 1: all low
        [1.2, 0.25, 1.00, 3.0],   # 2: all high
        [0.7, 0.10, 0.35, 1.5],   # 3: low speedFactor only
        [1.2, 0.10, 0.35, 1.5],   # 4: high speedFactor only
        [0.9, 0.10, 0.00, 1.5],   # 5: zero sigma (no imperfection)
        [0.9, 0.10, 1.00, 1.5],   # 6: max sigma (very imperfect)
        [0.9, 0.10, 0.35, 0.5],   # 7: low tau (aggressive following)
        [0.9, 0.10, 0.35, 3.0],   # 8: high tau (conservative following)
        [0.95, 0.15, 0.50, 1.0],  # 9: random mid-range
    ]

    theta_labels = [
        "default",
        "all_low",
        "all_high",
        "low_speedFactor",
        "high_speedFactor",
        "zero_sigma",
        "max_sigma",
        "low_tau",
        "high_tau",
        "random_mid",
    ]

    all_raw_dfs = []
    summary_rows = []

    for sim_id, (theta, label) in enumerate(zip(thetas, theta_labels)):
        theta = np.array(theta)
        print(f"  Sim {sim_id} ({label}): "
              f"speedFactor={theta[0]:.2f} speedDev={theta[1]:.2f} "
              f"sigma={theta[2]:.2f} tau={theta[3]:.2f} ... ", end="", flush=True)

        raw_df, elapsed = run_one_sim(sim_id, theta)

        if raw_df is not None:
            raw_df["sim_label"] = label
            all_raw_dfs.append(raw_df)
            summary_rows.append(compute_summary_row(sim_id, theta, raw_df))
            print(f"done in {elapsed:.1f}s "
                  f"({len(raw_df)} rows, "
                  f"mean speed={raw_df['meanSpeed'].mean():.2f} m/s)")
        else:
            print("FAILED")

    # === Save raw data ===
    if all_raw_dfs:
        raw_all = pd.concat(all_raw_dfs, ignore_index=True)
        raw_all.to_csv("inspect_raw_edgedata.csv", index=False)
        print(f"\nSaved raw data: inspect_raw_edgedata.csv")
        print(f"  Shape: {raw_all.shape}")
        print(f"  Columns: {list(raw_all.columns)}")

    # === Save summary (one row per simulation - THIS IS WHAT SBI USES) ===
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv("inspect_summary.csv", index=False)
        print(f"\nSaved summary: inspect_summary.csv")
        print(f"  Shape: {summary_df.shape}")
        print(f"  Columns: {list(summary_df.columns)}")

    # === Print comparison table ===
    print()
    print("=" * 60)
    print("COMPARISON: How parameters affect traffic")
    print("=" * 60)
    print()

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)

        # Print nice table
        print(f"{'Label':<20} {'sF':>5} {'sD':>5} {'sig':>5} {'tau':>5} | "
              f"{'MeanSpd':>8} {'StdSpd':>8} {'<5m/s':>6} {'Vehicles':>9}")
        print("-" * 90)

        for _, r in summary_df.iterrows():
            sim_id = int(r["sim_id"])
            label = theta_labels[sim_id]
            print(f"{label:<20} "
                  f"{r['theta_speedFactor']:5.2f} "
                  f"{r['theta_speedDev']:5.2f} "
                  f"{r['theta_sigma']:5.2f} "
                  f"{r['theta_tau']:5.2f} | "
                  f"{r['global_meanSpeed']:8.3f} "
                  f"{r['global_stdSpeed']:8.3f} "
                  f"{r['frac_below_5ms']:6.3f} "
                  f"{r['total_vehicles']:9.0f}")

    # === Quick sensitivity summary ===
    print()
    print("=" * 60)
    print("SENSITIVITY: Which parameters matter most?")
    print("=" * 60)
    print()

    if summary_rows:
        sd = pd.DataFrame(summary_rows)

        # speedFactor: compare sim 3 (low=0.7) vs sim 4 (high=1.2)
        if len(sd) >= 5:
            low_sf = sd[sd["sim_id"] == 3]["global_meanSpeed"].values[0]
            high_sf = sd[sd["sim_id"] == 4]["global_meanSpeed"].values[0]
            print(f"  speedFactor (0.7 vs 1.2): "
                  f"mean speed {low_sf:.2f} vs {high_sf:.2f} "
                  f"(diff = {high_sf - low_sf:+.2f} m/s)")

        # sigma: compare sim 5 (0.0) vs sim 6 (1.0)
        if len(sd) >= 7:
            low_sig = sd[sd["sim_id"] == 5]["global_meanSpeed"].values[0]
            high_sig = sd[sd["sim_id"] == 6]["global_meanSpeed"].values[0]
            print(f"  sigma      (0.0 vs 1.0): "
                  f"mean speed {low_sig:.2f} vs {high_sig:.2f} "
                  f"(diff = {high_sig - low_sig:+.2f} m/s)")

        # tau: compare sim 7 (0.5) vs sim 8 (3.0)
        if len(sd) >= 9:
            low_tau = sd[sd["sim_id"] == 7]["global_meanSpeed"].values[0]
            high_tau = sd[sd["sim_id"] == 8]["global_meanSpeed"].values[0]
            print(f"  tau        (0.5 vs 3.0): "
                  f"mean speed {low_tau:.2f} vs {high_tau:.2f} "
                  f"(diff = {high_tau - low_tau:+.2f} m/s)")

        # all low vs all high
        if len(sd) >= 3:
            all_low = sd[sd["sim_id"] == 1]["global_meanSpeed"].values[0]
            all_high = sd[sd["sim_id"] == 2]["global_meanSpeed"].values[0]
            print(f"  ALL params (low vs high): "
                  f"mean speed {all_low:.2f} vs {all_high:.2f} "
                  f"(diff = {all_high - all_low:+.2f} m/s)")

    print()
    print("Check the CSV files to explore the data:")
    print("  inspect_raw_edgedata.csv  - every edge, every time bin, every sim")
    print("  inspect_summary.csv       - one row per sim (what SBI sees)")


if __name__ == "__main__":
    main()