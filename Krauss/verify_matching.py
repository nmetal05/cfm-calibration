"""
Verify SUMO and TomTom produce matching summary stats — FINAL
"""
import numpy as np
import pandas as pd
import torch
from parse_output_shared import (
    parse_edge_data,
    summary_statistics_sumo,
    summary_statistics_tomtom,
    get_edge_max_speeds,
)

# ============================================================
print("=" * 70)
print("LOADING NETWORK DATA")
print("=" * 70)
edge_max_speeds, edge_lengths = get_edge_max_speeds("osm.net.xml")
print(f"  Edges with speed data: {len(edge_max_speeds)}")

# ============================================================
print("\n" + "=" * 70)
print("COMPUTING x_sim (SUMO)")
print("=" * 70)
sumo_data = parse_edge_data("debug_sim/edgedata.xml")
x_sim = summary_statistics_sumo(sumo_data, edge_max_speeds, edge_lengths)
print(f"  x_sim shape: {x_sim.shape}")

# ============================================================
print("\n" + "=" * 70)
print("COMPUTING x_obs (TomTom)")
print("=" * 70)
tomtom = pd.read_csv("tomtom_all_data.csv")
evening = tomtom[tomtom["timeSet"] == 4]
x_obs = summary_statistics_tomtom(evening, min_samples=30)
print(f"  x_obs shape: {x_obs.shape}")

# ============================================================
print("\n" + "=" * 70)
print("SHAPE CHECK")
print("=" * 70)
if x_sim.shape == x_obs.shape:
    print(f"  ✅ MATCH! Both have {x_sim.shape[0]} dimensions")
else:
    print(f"  ❌ MISMATCH! SUMO={x_sim.shape}, TomTom={x_obs.shape}")
    exit(1)

# ============================================================
print("\n" + "=" * 70)
print("FULL COMPARISON")
print("=" * 70)

stat_names = (
    [f"speedP{p}" for p in [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]] +
    ["mean_speed", "std_speed", "median_speed", "iqr_speed"] +
    [f"stdP{p}" for p in [10,25,50,75,90]] + ["mean_std"] +
    [f"bin_{a}-{b}" for a,b in zip([0,5,10,15,20,25,30,35,40,50,60],[5,10,15,20,25,30,35,40,50,60,120])] +
    [f"ttr_P{p}" for p in [10,25,50,75,90,95]] +
    ["ttr_mean", "ttr_std", "ttr_median",
     "congestion>10%", "congestion>25%", "congestion>50%"] +
    ["frac<10", "frac<15", "frac<20", "frac>40", "frac>50"]
)

# Print in sections
sections = [
    ("SPEED PERCENTILES", 0, 19),
    ("SPEED STATS", 19, 23),
    ("SPEED VARIABILITY", 23, 29),
    ("SPEED HISTOGRAM", 29, 40),
    ("CONGESTION (TTR)", 40, 52),
    ("SPEED THRESHOLDS", 52, 57),
]

for section_name, start, end in sections:
    print(f"\n  --- {section_name} ---")
    print(f"  {'Statistic':<25s} {'SUMO':>10s} {'TomTom':>10s} {'Diff':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for i in range(start, min(end, len(x_sim), len(x_obs))):
        name = stat_names[i] if i < len(stat_names) else f"stat_{i}"
        print(f"  {name:<25s} {x_sim[i]:10.3f} {x_obs[i]:10.3f} {x_sim[i]-x_obs[i]:+10.3f}")

# ============================================================
print("\n" + "=" * 70)
print("SUMMARY FOR YOUR PRESENTATION")
print("=" * 70)

mean_idx = 19
ttr_mean_idx = stat_names.index("ttr_mean")
cong10_idx = stat_names.index("congestion>10%")
cong25_idx = stat_names.index("congestion>25%")
cong50_idx = stat_names.index("congestion>50%")

print(f"""
  SPEED COMPARISON:
    SUMO default:     {x_sim[mean_idx]:.1f} km/h
    TomTom real:      {x_obs[mean_idx]:.1f} km/h
    Gap:              {x_sim[mean_idx]-x_obs[mean_idx]:+.1f} km/h
    → SBI will adjust speedFactor and sigma to close this gap

  CONGESTION COMPARISON (Travel Time Ratio):
    SUMO default TTR:   {x_sim[ttr_mean_idx]:.3f}  (congestion = {(x_sim[ttr_mean_idx]-1)*100:.1f}%)
    TomTom real TTR:    {x_obs[ttr_mean_idx]:.3f}  (congestion = {(x_obs[ttr_mean_idx]-1)*100:.1f}%)

    SUMO: {x_sim[cong10_idx]*100:.1f}% of edges have >10% delay
    Real: {x_obs[cong10_idx]*100:.1f}% of segments have >10% delay

    SUMO: {x_sim[cong50_idx]*100:.1f}% of edges have >50% delay
    Real: {x_obs[cong50_idx]*100:.1f}% of segments have >50% delay

  → SBI calibrates BOTH speed and congestion simultaneously!
  → After calibration, compare Krauss vs IDM vs Wiedemann
     on which model best matches BOTH metrics
""")

# Save
x_obs_tensor = torch.tensor(x_obs, dtype=torch.float32)
torch.save(x_obs_tensor, "x_obs.pt")
print("Saved: x_obs.pt (57 dimensions: speed + congestion)")

with open("stat_names.txt", "w") as f:
    for name in stat_names:
        f.write(name + "\n")
print("Saved: stat_names.txt")