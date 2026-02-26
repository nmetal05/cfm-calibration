"""
Investigate TomTom's travelTimeRatio and congestion definition
"""
import pandas as pd
import numpy as np

tomtom = pd.read_csv("tomtom_all_data.csv")
evening = tomtom[tomtom["timeSet"] == 4].copy()
evening_good = evening[evening["sampleSize"] >= 30].copy()

print("=" * 70)
print("INVESTIGATING TOMTOM TRAVEL TIME RATIO")
print("=" * 70)

ttr = evening_good["travelTimeRatio"].dropna()
print(f"\n  Segments with TTR data: {len(ttr)}")
print(f"  Mean:   {ttr.mean():.3f}")
print(f"  Median: {ttr.median():.3f}")
print(f"  Min:    {ttr.min():.3f}")
print(f"  Max:    {ttr.max():.3f}")
print(f"  Std:    {ttr.std():.3f}")

print(f"\n  Distribution:")
print(f"    TTR < 0.5:   {(ttr < 0.5).mean():.3f}  ({(ttr < 0.5).sum()} segments)")
print(f"    TTR 0.5-1.0: {((ttr >= 0.5) & (ttr < 1.0)).mean():.3f}")
print(f"    TTR = 1.0:   {(ttr == 1.0).mean():.3f}")
print(f"    TTR 1.0-1.5: {((ttr > 1.0) & (ttr <= 1.5)).mean():.3f}")
print(f"    TTR 1.5-2.0: {((ttr > 1.5) & (ttr <= 2.0)).mean():.3f}")
print(f"    TTR > 2.0:   {(ttr > 2.0).mean():.3f}")

# ============================================================
print("\n" + "=" * 70)
print("CHECK: Does TTR relate to speed correctly?")
print("=" * 70)

# If TTR = actual/freeflow, then high TTR = slow (congested)
# If TTR < 1, that's weird unless TomTom defines it differently

# Let's check relationship between speed and TTR
eg = evening_good[["averageSpeed", "speedLimit", "travelTimeRatio",
                     "averageTravelTime", "distance"]].dropna()

print(f"\n  Checking speed vs TTR relationship:")
print(f"  Segments with all data: {len(eg)}")

# Compute what TTR SHOULD be based on speed and speed limit
eg["expected_ttr"] = eg["speedLimit"] / eg["averageSpeed"].clip(lower=0.1)
# Or maybe it's the other way: actual_time / freeflow_time
eg["freeflow_time"] = eg["distance"] / (eg["speedLimit"] / 3.6)  # seconds
eg["actual_ttr_from_time"] = eg["averageTravelTime"] / eg["freeflow_time"].clip(lower=0.01)

print(f"\n  TomTom TTR vs computed TTR:")
print(f"    TomTom TTR mean:              {eg['travelTimeRatio'].mean():.3f}")
print(f"    speedLimit/avgSpeed mean:      {eg['expected_ttr'].mean():.3f}")
print(f"    avgTravelTime/freeflowTime:    {eg['actual_ttr_from_time'].mean():.3f}")

# Correlation
corr1 = eg["travelTimeRatio"].corr(eg["expected_ttr"])
corr2 = eg["travelTimeRatio"].corr(eg["actual_ttr_from_time"])
print(f"\n  Correlation with TomTom TTR:")
print(f"    vs speedLimit/avgSpeed:        {corr1:.3f}")
print(f"    vs avgTravelTime/freeflowTime: {corr2:.3f}")

# Show some examples
print(f"\n  Sample segments:")
print(f"  {'segID':>20s} {'avgSpd':>7s} {'spdLim':>7s} {'TTR':>7s} {'dist':>7s} {'avgTT':>7s} {'freeflTT':>8s} {'compTTR':>8s}")
print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")

for _, row in eg.head(15).iterrows():
    print(f"  {row['averageSpeed']:20.1f} {row['averageSpeed']:7.1f} {row['speedLimit']:7.0f} "
          f"{row['travelTimeRatio']:7.2f} {row['distance']:7.1f} "
          f"{row['averageTravelTime']:7.2f} {row['freeflow_time']:8.2f} "
          f"{row['actual_ttr_from_time']:8.2f}")

# ============================================================
print("\n" + "=" * 70)
print("COMPUTE CONGESTION LEVEL (TomTom's definition)")
print("=" * 70)

# TomTom definition: congestion = (actual_time/freeflow_time - 1) * 100
# We need to figure out which field gives us this

# Option 1: use travelTimeRatio directly if it's actual/freeflow
# Option 2: compute from averageTravelTime and distance/speedLimit

# Compute from scratch
eg["congestion_pct"] = (eg["actual_ttr_from_time"] - 1.0) * 100

print(f"\n  Congestion level (computed from avgTravelTime / freeflowTime):")
print(f"    Mean:   {eg['congestion_pct'].mean():.1f}%")
print(f"    Median: {eg['congestion_pct'].median():.1f}%")
print(f"    Min:    {eg['congestion_pct'].min():.1f}%")
print(f"    Max:    {eg['congestion_pct'].max():.1f}%")

print(f"\n  Congestion distribution:")
print(f"    Free flow (< 10%):    {(eg['congestion_pct'] < 10).mean():.3f}")
print(f"    Light (10-25%):       {((eg['congestion_pct'] >= 10) & (eg['congestion_pct'] < 25)).mean():.3f}")
print(f"    Moderate (25-50%):    {((eg['congestion_pct'] >= 25) & (eg['congestion_pct'] < 50)).mean():.3f}")
print(f"    Heavy (50-100%):      {((eg['congestion_pct'] >= 50) & (eg['congestion_pct'] < 100)).mean():.3f}")
print(f"    Severe (> 100%):      {(eg['congestion_pct'] >= 100).mean():.3f}")

# ============================================================
print("\n" + "=" * 70)
print("BY ROAD CLASS")
print("=" * 70)

frc_names = {
    1: "Major road", 2: "Other major", 3: "Secondary",
    4: "Local connecting", 5: "Local high", 6: "Local", 7: "Local minor",
}

eg_frc = evening_good.merge(
    eg[["congestion_pct"]],
    left_index=True, right_index=True, how="inner"
)

# Actually let's just recompute for all good segments
full = evening_good.copy()
full["freeflow_time"] = full["distance"] / (full["speedLimit"] / 3.6).clip(lower=0.01)
full["congestion_pct"] = (full["averageTravelTime"] / full["freeflow_time"].clip(lower=0.01) - 1) * 100

for frc in sorted(full["frc"].dropna().unique()):
    frc_data = full[full["frc"] == frc]["congestion_pct"].dropna()
    if len(frc_data) > 0:
        name = frc_names.get(int(frc), "?")
        print(f"  FRC {int(frc)} ({name:18s}): "
              f"n={len(frc_data):5d}  "
              f"mean={frc_data.mean():6.1f}%  "
              f"median={frc_data.median():6.1f}%  "
              f"frac>50%={( frc_data > 50).mean():.3f}")

# ============================================================
print("\n" + "=" * 70)
print("WHAT TO USE FOR SBI")
print("=" * 70)

print(f"""
  For SBI summary statistics, we should use:
  
  1. SPEED distribution (what we already have)
     - Percentiles, mean, std, histogram
     
  2. CONGESTION distribution (NEW)
     - congestion_pct = (avgTravelTime / freeflowTime - 1) * 100
     - For SUMO: compute freeflowTime from edge length / speed limit
     - Percentiles of congestion across segments
     - Fraction in each congestion level

  The key insight: 
    TomTom's travelTimeRatio field might NOT be actual/freeflow directly.
    Computing congestion from avgTravelTime and distance/speedLimit is safer.
""")