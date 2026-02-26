"""
Analyze the SUMO vs TomTom mismatch
Figure out which segments are comparable
Design the right summary statistics for SBI

Usage: python analyze_mismatch.py
"""

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from parse_output import parse_edge_data

# ============================================================
print("=" * 70)
print("PART 1: ANALYZE TOMTOM DATA QUALITY")
print("=" * 70)

tomtom = pd.read_csv("tomtom_all_data.csv")
evening = tomtom[tomtom["timeSet"] == 4].copy()

print(f"\nAll evening segments: {len(evening)}")

# Filter by sample size
for min_samples in [0, 1, 5, 10, 30, 50, 100]:
    valid = evening[evening["sampleSize"] >= min_samples]
    valid_speeds = valid["averageSpeed"].dropna()
    if len(valid_speeds) > 0:
        print(f"\n  sampleSize >= {min_samples:4d}: "
              f"{len(valid_speeds):5d} segments  "
              f"mean={valid_speeds.mean():5.1f} km/h ({valid_speeds.mean()/3.6:5.2f} m/s)  "
              f"median={valid_speeds.median():5.1f} km/h")

print(f"\n  → Filtering to sampleSize >= 30 removes unreliable segments")

# ============================================================
print("\n" + "=" * 70)
print("PART 2: FILTERED EVENING DATA BY FRC")
print("=" * 70)

MIN_SAMPLES = 30
evening_good = evening[evening["sampleSize"] >= MIN_SAMPLES].copy()
print(f"\nFiltered evening segments (sampleSize >= {MIN_SAMPLES}): {len(evening_good)}")

frc_names = {
    0: "Motorway", 1: "Major road", 2: "Other major", 3: "Secondary",
    4: "Local connecting", 5: "Local high", 6: "Local", 7: "Local minor",
}

print(f"\n  FRC breakdown after filtering:")
for frc in sorted(evening_good["frc"].dropna().unique()):
    frc_data = evening_good[evening_good["frc"] == frc]
    speeds = frc_data["averageSpeed"].dropna()
    name = frc_names.get(int(frc), "?")
    print(f"    FRC {int(frc)} ({name:18s}): "
          f"n={len(speeds):5d}  "
          f"mean={speeds.mean():5.1f} km/h ({speeds.mean()/3.6:5.2f} m/s)  "
          f"std={speeds.std():5.1f}  "
          f"median={speeds.median():5.1f}")

# Overall stats with filtering
valid = evening_good["averageSpeed"].dropna()
print(f"\n  After filtering:")
print(f"    Mean:   {valid.mean():.1f} km/h = {valid.mean()/3.6:.2f} m/s")
print(f"    Median: {valid.median():.1f} km/h = {valid.median()/3.6:.2f} m/s")
print(f"    Std:    {valid.std():.1f} km/h = {valid.std()/3.6:.2f} m/s")

# ============================================================
print("\n" + "=" * 70)
print("PART 3: ANALYZE SUMO NETWORK — EDGE TYPES AND SPEEDS")
print("=" * 70)

# Parse SUMO edgedata from our debug run
edgedata_path = Path("debug_sim/edgedata.xml")
if not edgedata_path.exists():
    # Try inspect runs
    edgedata_path = Path("inspect_runs/sim_00/edgedata.xml")

if edgedata_path.exists():
    print(f"\nParsing SUMO edgedata from: {edgedata_path}")
    tree = ET.parse(str(edgedata_path))
    root = tree.getroot()

    # Collect per-edge average speed across ALL time bins
    edge_speeds = {}
    edge_counts = {}
    for interval in root.findall("interval"):
        for edge in interval.findall("edge"):
            eid = edge.get("id")
            speed = float(edge.get("speed", 0))
            entered = int(edge.get("entered", 0))
            if eid not in edge_speeds:
                edge_speeds[eid] = []
                edge_counts[eid] = 0
            edge_speeds[eid].append(speed)
            edge_counts[eid] += entered

    # Compute mean speed per edge across all time bins
    sumo_edges = pd.DataFrame([
        {
            "edge_id": eid,
            "mean_speed_ms": np.mean(speeds),
            "std_speed_ms": np.std(speeds),
            "mean_speed_kmh": np.mean(speeds) * 3.6,
            "total_vehicles": edge_counts[eid],
        }
        for eid, speeds in edge_speeds.items()
    ])

    print(f"  SUMO edges: {len(sumo_edges)}")
    print(f"  SUMO mean speed: {sumo_edges['mean_speed_kmh'].mean():.1f} km/h "
          f"({sumo_edges['mean_speed_ms'].mean():.2f} m/s)")
    print(f"  SUMO median speed: {sumo_edges['mean_speed_kmh'].median():.1f} km/h")
    print(f"  SUMO std speed: {sumo_edges['mean_speed_kmh'].std():.1f} km/h")

    # Filter SUMO to edges with actual traffic
    sumo_with_traffic = sumo_edges[sumo_edges["total_vehicles"] > 0]
    print(f"\n  SUMO edges with traffic: {len(sumo_with_traffic)}")
    print(f"  Mean speed (with traffic): {sumo_with_traffic['mean_speed_kmh'].mean():.1f} km/h "
          f"({sumo_with_traffic['mean_speed_ms'].mean():.2f} m/s)")
    print(f"  Median speed (with traffic): {sumo_with_traffic['mean_speed_kmh'].median():.1f} km/h")

    # Speed distribution
    print(f"\n  SUMO speed percentiles (edges with traffic, km/h):")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        val = sumo_with_traffic["mean_speed_kmh"].quantile(p / 100)
        print(f"    P{p:2d}: {val:5.1f} km/h ({val/3.6:5.2f} m/s)")

    # ============================================================
    print("\n" + "=" * 70)
    print("PART 4: SIDE-BY-SIDE SPEED DISTRIBUTIONS")
    print("=" * 70)

    # TomTom (filtered)
    tt_speeds = evening_good["averageSpeed"].dropna().values
    # SUMO (edges with traffic)
    sumo_speeds = sumo_with_traffic["mean_speed_kmh"].values

    print(f"\n  {'Statistic':<25s} {'TomTom':>12s} {'SUMO':>12s} {'Diff':>12s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")

    stats = [
        ("Count", len(tt_speeds), len(sumo_speeds)),
        ("Mean (km/h)", np.mean(tt_speeds), np.mean(sumo_speeds)),
        ("Median (km/h)", np.median(tt_speeds), np.median(sumo_speeds)),
        ("Std (km/h)", np.std(tt_speeds), np.std(sumo_speeds)),
        ("P5 (km/h)", np.percentile(tt_speeds, 5), np.percentile(sumo_speeds, 5)),
        ("P10 (km/h)", np.percentile(tt_speeds, 10), np.percentile(sumo_speeds, 10)),
        ("P25 (km/h)", np.percentile(tt_speeds, 25), np.percentile(sumo_speeds, 25)),
        ("P50 (km/h)", np.percentile(tt_speeds, 50), np.percentile(sumo_speeds, 50)),
        ("P75 (km/h)", np.percentile(tt_speeds, 75), np.percentile(sumo_speeds, 75)),
        ("P90 (km/h)", np.percentile(tt_speeds, 90), np.percentile(sumo_speeds, 90)),
        ("P95 (km/h)", np.percentile(tt_speeds, 95), np.percentile(sumo_speeds, 95)),
        ("Frac < 10 km/h", (tt_speeds < 10).mean(), (sumo_speeds < 10).mean()),
        ("Frac < 20 km/h", (tt_speeds < 20).mean(), (sumo_speeds < 20).mean()),
        ("Frac > 40 km/h", (tt_speeds > 40).mean(), (sumo_speeds > 40).mean()),
    ]

    for name, tt_val, sumo_val in stats:
        if isinstance(tt_val, int):
            print(f"  {name:<25s} {tt_val:>12d} {sumo_val:>12d} {sumo_val-tt_val:>12d}")
        else:
            print(f"  {name:<25s} {tt_val:>12.2f} {sumo_val:>12.2f} {sumo_val-tt_val:>+12.2f}")

    # ============================================================
    print("\n" + "=" * 70)
    print("PART 5: MATCHING STRATEGY")
    print("=" * 70)

    # Try filtering TomTom to only major roads (FRC <= 5) to match SUMO
    for max_frc in [3, 4, 5, 6, 7]:
        tt_filtered = evening_good[evening_good["frc"] <= max_frc]["averageSpeed"].dropna()
        if len(tt_filtered) > 0:
            print(f"\n  TomTom FRC <= {max_frc}: "
                  f"n={len(tt_filtered):5d}  "
                  f"mean={tt_filtered.mean():5.1f} km/h ({tt_filtered.mean()/3.6:5.2f} m/s)  "
                  f"median={tt_filtered.median():5.1f} km/h")

    print(f"\n  SUMO all edges with traffic: "
          f"n={len(sumo_with_traffic)}  "
          f"mean={sumo_with_traffic['mean_speed_kmh'].mean():.1f} km/h "
          f"({sumo_with_traffic['mean_speed_ms'].mean():.2f} m/s)")

else:
    print("\n  ❌ No SUMO edgedata found. Run debug_fixed.py first.")

# ============================================================
print("\n" + "=" * 70)
print("PART 6: WHAT SUMMARY STATS CAN WE USE FOR SBI?")
print("=" * 70)

print("""
  PROBLEM: TomTom gives us ONE aggregate per segment for 16:00-21:00
           SUMO gives us per-5-min-bin data for 17:00-19:00
           
  SOLUTION: Compute COMPARABLE statistics from both:
  
  From TomTom (evening, filtered):
    - Distribution of averageSpeed across segments
    - Distribution of standardDeviationSpeed across segments  
    - Distribution of travelTimeRatio across segments
    - Speed percentiles (network-wide)
    - Statistics by FRC class
    
  From SUMO (aggregated over full simulation):
    - Distribution of mean speed across edges (averaged over all time bins)
    - Distribution of speed std across edges
    - Speed percentiles (network-wide)
    
  Both give us: spatial distribution of speeds across road segments
  
  Summary stats vector will contain:
    - Network-wide speed percentiles (P5 through P95)
    - Mean and std of speed
    - Fraction in various speed bins
    - Mean speed by road class (if matchable)
""")

# Save filtered evening data for next steps
evening_good.to_csv("tomtom_evening_filtered.csv", index=False)
print(f"Saved: tomtom_evening_filtered.csv ({len(evening_good)} segments)")

print()
print("=" * 70)
print("Share this output — I'll design the final matching strategy")
print("=" * 70)