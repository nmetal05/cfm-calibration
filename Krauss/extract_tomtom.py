"""
Extract and analyze TomTom data for SBI — FIXED
Usage: python extract_tomtom.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

print("Loading TomTom JSON (63 MB, may take a moment)...")
with open("tomtom_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ============================================================
print("\n" + "=" * 70)
print("SECTION 1: TIME PERIODS")
print("=" * 70)

time_sets = data["timeSets"]
time_set_map = {}
for ts in time_sets:
    ts_id = ts["@id"]
    name = ts["name"]
    time_set_map[ts_id] = name
    print(f"  TimeSet @id={ts_id}: '{name}'")

# Evening peak is @id=4: '16:00-21:00'
EVENING_ID = 4
print(f"\n  Using evening peak: @id={EVENING_ID} '{time_set_map[EVENING_ID]}'")

# ============================================================
print("\n" + "=" * 70)
print("SECTION 2: INSPECT SPEED PERCENTILES FORMAT")
print("=" * 70)

# Check what speedPercentiles actually looks like
first_seg = data["network"]["segmentResults"][0]
first_tr = first_seg["segmentTimeResults"][0]
pcts = first_tr.get("speedPercentiles", [])
print(f"\n  speedPercentiles type: {type(pcts)}")
print(f"  speedPercentiles length: {len(pcts)}")
print(f"  speedPercentiles content: {pcts}")
if len(pcts) > 0:
    print(f"  First element type: {type(pcts[0])}")
    print(f"  First element value: {pcts[0]}")

# ============================================================
print("\n" + "=" * 70)
print("SECTION 3: EXTRACT ALL SEGMENT DATA")
print("=" * 70)

segments = data["network"]["segmentResults"]
print(f"\nTotal segments: {len(segments)}")

rows = []
for seg in segments:
    seg_base = {
        "segmentId": seg["segmentId"],
        "newSegmentId": seg.get("newSegmentId", ""),
        "speedLimit": seg.get("speedLimit", None),
        "frc": seg.get("frc", None),
        "distance": seg.get("distance", None),
    }

    # Get lat/lon of first point
    if "shape" in seg and len(seg["shape"]) > 0:
        seg_base["lat"] = seg["shape"][0].get("latitude", None)
        seg_base["lon"] = seg["shape"][0].get("longitude", None)

    for tr in seg.get("segmentTimeResults", []):
        row = seg_base.copy()
        row["timeSet"] = tr.get("timeSet", None)
        row["timeSetName"] = time_set_map.get(tr.get("timeSet"), "unknown")
        row["dateRange"] = tr.get("dateRange", None)
        row["averageSpeed"] = tr.get("averageSpeed", None)
        row["medianSpeed"] = tr.get("medianSpeed", None)
        row["harmonicAverageSpeed"] = tr.get("harmonicAverageSpeed", None)
        row["standardDeviationSpeed"] = tr.get("standardDeviationSpeed", None)
        row["averageTravelTime"] = tr.get("averageTravelTime", None)
        row["medianTravelTime"] = tr.get("medianTravelTime", None)
        row["travelTimeRatio"] = tr.get("travelTimeRatio", None)
        row["sampleSize"] = tr.get("sampleSize", None)
        row["travelTimeStandardDeviation"] = tr.get("travelTimeStandardDeviation", None)

        # Handle speed percentiles - could be list of numbers or list of dicts
        percentiles = tr.get("speedPercentiles", [])
        if len(percentiles) > 0:
            if isinstance(percentiles[0], (int, float)):
                # It's just a list of speed values at fixed percentiles
                # TomTom typically uses: 5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95
                pct_labels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
                for j, spd in enumerate(percentiles):
                    if j < len(pct_labels):
                        row[f"speedP{pct_labels[j]}"] = spd
                    else:
                        row[f"speedP_idx{j}"] = spd
            elif isinstance(percentiles[0], dict):
                # It's a list of {percentile: X, speed: Y}
                for p in percentiles:
                    pct = p.get("percentile", None)
                    spd = p.get("speed", None)
                    if pct is not None:
                        row[f"speedP{pct}"] = spd

        rows.append(row)

df = pd.DataFrame(rows)
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

df.to_csv("tomtom_all_data.csv", index=False)
print(f"Saved: tomtom_all_data.csv")

# ============================================================
print("\n" + "=" * 70)
print("SECTION 4: STATISTICS PER TIME PERIOD")
print("=" * 70)

for ts_id in sorted(df["timeSet"].dropna().unique()):
    subset = df[df["timeSet"] == ts_id]
    name = time_set_map.get(int(ts_id), "unknown")
    valid_speeds = subset["averageSpeed"].dropna()

    print(f"\n  TimeSet {int(ts_id)}: '{name}'")
    print(f"    Segments with data: {len(valid_speeds)} / {len(subset)}")
    if len(valid_speeds) > 0:
        print(f"    Average speed (km/h):  mean={valid_speeds.mean():.1f}  "
              f"median={valid_speeds.median():.1f}  "
              f"std={valid_speeds.std():.1f}  "
              f"range=[{valid_speeds.min():.1f}, {valid_speeds.max():.1f}]")
        print(f"    Average speed (m/s):   mean={valid_speeds.mean()/3.6:.2f}  "
              f"median={valid_speeds.median()/3.6:.2f}")

        valid_samples = subset["sampleSize"].dropna()
        if len(valid_samples) > 0:
            print(f"    Sample sizes:          mean={valid_samples.mean():.0f}  "
                  f"median={valid_samples.median():.0f}  "
                  f"range=[{valid_samples.min():.0f}, {valid_samples.max():.0f}]")

# ============================================================
print("\n" + "=" * 70)
print("SECTION 5: EVENING PEAK (16:00-21:00) DETAIL")
print("=" * 70)

evening = df[df["timeSet"] == EVENING_ID].copy()
valid_evening = evening["averageSpeed"].dropna()

print(f"\n  Segments with evening data: {len(valid_evening)}")
print(f"\n  Speed statistics:")
print(f"    Mean:    {valid_evening.mean():.1f} km/h = {valid_evening.mean()/3.6:.2f} m/s")
print(f"    Median:  {valid_evening.median():.1f} km/h = {valid_evening.median()/3.6:.2f} m/s")
print(f"    Std:     {valid_evening.std():.1f} km/h = {valid_evening.std()/3.6:.2f} m/s")
print(f"    Min:     {valid_evening.min():.1f} km/h")
print(f"    Max:     {valid_evening.max():.1f} km/h")

# Harmonic average speed
valid_harmonic = evening["harmonicAverageSpeed"].dropna()
if len(valid_harmonic) > 0:
    print(f"\n  Harmonic average speed:")
    print(f"    Mean:    {valid_harmonic.mean():.1f} km/h = {valid_harmonic.mean()/3.6:.2f} m/s")

# Standard deviation of speed
valid_std = evening["standardDeviationSpeed"].dropna()
if len(valid_std) > 0:
    print(f"\n  Speed standard deviation (driver heterogeneity):")
    print(f"    Mean std: {valid_std.mean():.1f} km/h = {valid_std.mean()/3.6:.2f} m/s")

# Travel time ratio (congestion indicator)
valid_ratio = evening["travelTimeRatio"].dropna()
if len(valid_ratio) > 0:
    print(f"\n  Travel time ratio (1.0 = free flow, >1 = congested):")
    print(f"    Mean:   {valid_ratio.mean():.2f}")
    print(f"    Median: {valid_ratio.median():.2f}")
    print(f"    Max:    {valid_ratio.max():.2f}")
    print(f"    Frac > 1.5: {(valid_ratio > 1.5).mean():.3f}")
    print(f"    Frac > 2.0: {(valid_ratio > 2.0).mean():.3f}")

# FRC breakdown for evening
print(f"\n  Speed by road class (evening peak):")
frc_names = {
    0: "Motorway", 1: "Major road", 2: "Other major", 3: "Secondary",
    4: "Local connecting", 5: "Local high", 6: "Local", 7: "Local minor",
}
for frc in sorted(evening["frc"].dropna().unique()):
    frc_data = evening[evening["frc"] == frc]["averageSpeed"].dropna()
    if len(frc_data) > 0:
        name = frc_names.get(int(frc), "?")
        print(f"    FRC {int(frc)} ({name:18s}): "
              f"n={len(frc_data):5d}  "
              f"mean={frc_data.mean():5.1f} km/h  "
              f"({frc_data.mean()/3.6:5.2f} m/s)")

# ============================================================
print("\n" + "=" * 70)
print("SECTION 6: SPEED PERCENTILES (EVENING)")
print("=" * 70)

pct_cols = sorted([c for c in df.columns if c.startswith("speedP")])
if pct_cols:
    print(f"\n  Percentile columns found: {pct_cols}")
    evening_pcts = evening[pct_cols].mean()
    print(f"\n  Network-average speed percentiles (evening, km/h):")
    for col in pct_cols:
        val = evening_pcts[col]
        if pd.notna(val):
            print(f"    {col:>10s}: {val:6.1f} km/h = {val/3.6:5.2f} m/s")

# ============================================================
print("\n" + "=" * 70)
print("SECTION 7: COMPARISON WITH SUMO")
print("=" * 70)

print(f"""
  SUMO evening peak simulation (default type_3):
    Mean speed across all edges:  10.1 m/s = 36.4 km/h
    Std speed across edges:        2.92 m/s = 10.5 km/h

  TomTom evening peak (16:00-21:00):
    Mean speed across segments:   {valid_evening.mean()/3.6:.2f} m/s = {valid_evening.mean():.1f} km/h
    Std speed across segments:    {valid_evening.std()/3.6:.2f} m/s = {valid_evening.std():.1f} km/h

  Difference: {abs(10.1 - valid_evening.mean()/3.6):.2f} m/s
""")

# ============================================================
print("\n" + "=" * 70)
print("SECTION 8: SAMPLE SEGMENTS")
print("=" * 70)

sample_segs = evening.nlargest(3, "sampleSize")["segmentId"].values
for seg_id in sample_segs:
    seg_data = evening[evening["segmentId"] == seg_id].iloc[0]
    print(f"\n  Segment {seg_id}:")
    print(f"    Speed limit:  {seg_data.get('speedLimit', '?')} km/h")
    print(f"    FRC:          {seg_data.get('frc', '?')}")
    print(f"    Distance:     {seg_data.get('distance', '?')} m")
    print(f"    Avg speed:    {seg_data.get('averageSpeed', '?')} km/h")
    print(f"    Median speed: {seg_data.get('medianSpeed', '?')} km/h")
    print(f"    Std speed:    {seg_data.get('standardDeviationSpeed', '?')} km/h")
    print(f"    Travel time:  {seg_data.get('averageTravelTime', '?')} s")
    print(f"    TT ratio:     {seg_data.get('travelTimeRatio', '?')}")
    print(f"    Sample size:  {seg_data.get('sampleSize', '?')}")

print()
print("=" * 70)
print("Share this output with me and I'll build the x_obs extraction!")
print("=" * 70)