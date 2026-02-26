"""
Diagnose the mismatch between x_obs and training simulations.
Figure out WHY x_obs is out of range and how to fix it.

Usage: python diagnose_mismatch.py
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PARAM_NAMES = ["speedFactor", "speedDev", "sigma", "tau"]

# Load stat names if available
stat_names = []
if Path("stat_names.txt").exists():
    with open("stat_names.txt") as f:
        stat_names = [line.strip() for line in f if line.strip()]

# Load data
theta = torch.load("sbi_thetas.pt", weights_only=True).numpy()
x = torch.load("sbi_xs.pt", weights_only=True).numpy()
x_obs = torch.load("x_obs.pt", weights_only=True).numpy()

print("=" * 70)
print("  MISMATCH DIAGNOSTIC")
print("=" * 70)
print(f"Training: {x.shape[0]} sims, {x.shape[1]} features")
print(f"x_obs:    {x_obs.shape[0]} features")
print()

# ============================================================
# Per-feature analysis
# ============================================================
print(f"{'Feat':>4} | {'Name':>30} | {'x_obs':>10} | "
      f"{'Train Min':>10} | {'Train Max':>10} | {'Train Mean':>10} | {'Status':>10}")
print("-" * 110)

out_of_range = []
for j in range(x.shape[1]):
    name = stat_names[j] if j < len(stat_names) else f"feat_{j}"
    obs_val = x_obs[j]
    t_min = x[:, j].min()
    t_max = x[:, j].max()
    t_mean = x[:, j].mean()
    t_std = x[:, j].std()

    if obs_val < t_min:
        status = "BELOW"
        distance = (t_min - obs_val) / max(t_std, 1e-8)
        out_of_range.append((j, name, obs_val, t_min, t_max, t_mean, "BELOW", distance))
    elif obs_val > t_max:
        status = "ABOVE"
        distance = (obs_val - t_max) / max(t_std, 1e-8)
        out_of_range.append((j, name, obs_val, t_min, t_max, t_mean, "ABOVE", distance))
    else:
        status = "OK"
        distance = 0

    # Print all, highlight problems
    marker = "***" if status != "OK" else "   "
    print(f"{j:4d} | {name:>30} | {obs_val:10.4f} | "
          f"{t_min:10.4f} | {t_max:10.4f} | {t_mean:10.4f} | "
          f"{status:>6} {marker} ({distance:.1f}σ)")

print(f"\n--- Summary ---")
print(f"Total features:      {x.shape[1]}")
print(f"In range:            {x.shape[1] - len(out_of_range)}")
print(f"Out of range:        {len(out_of_range)}")

if out_of_range:
    above = [r for r in out_of_range if r[6] == "ABOVE"]
    below = [r for r in out_of_range if r[6] == "BELOW"]
    print(f"  Above train max:   {len(above)}")
    print(f"  Below train min:   {len(below)}")

    print(f"\n--- Worst Offenders (sorted by distance in std devs) ---")
    out_of_range.sort(key=lambda r: r[7], reverse=True)
    for j, name, obs_val, t_min, t_max, t_mean, direction, dist in out_of_range[:10]:
        print(f"  feat {j:2d} ({name:>25}): x_obs={obs_val:.4f}, "
              f"train=[{t_min:.4f}, {t_max:.4f}], {direction} by {dist:.1f}σ")

# ============================================================
# Check what x_obs was generated from
# ============================================================
print("\n" + "=" * 70)
print("  INVESTIGATING x_obs ORIGIN")
print("=" * 70)

# Check if x_obs matches any training sim closely
distances = np.sqrt(((x - x_obs) ** 2).sum(axis=1))
closest_idx = distances.argmin()
closest_dist = distances[closest_idx]
mean_dist = distances.mean()

print(f"Closest training sim to x_obs: idx={closest_idx}, dist={closest_dist:.4f}")
print(f"Mean distance to x_obs:        {mean_dist:.4f}")
print(f"Closest sim params: {dict(zip(PARAM_NAMES, theta[closest_idx]))}")

# Normalized distances (per feature)
x_std = x.std(axis=0)
x_std[x_std < 1e-8] = 1e-8
z_obs = (x_obs - x.mean(axis=0)) / x_std
print(f"\nZ-scored x_obs stats:")
print(f"  Mean |z|:  {np.abs(z_obs).mean():.2f}")
print(f"  Max |z|:   {np.abs(z_obs).max():.2f} (feat {np.abs(z_obs).argmax()})")
print(f"  Features with |z| > 3: {(np.abs(z_obs) > 3).sum()}")
print(f"  Features with |z| > 5: {(np.abs(z_obs) > 5).sum()}")

# ============================================================
# Group analysis: speed features vs congestion features
# ============================================================
print("\n" + "=" * 70)
print("  PATTERN ANALYSIS")
print("=" * 70)

# Try to identify groups from stat names
speed_feats = []
congestion_feats = []
for j in range(len(stat_names)):
    name = stat_names[j].lower() if j < len(stat_names) else ""
    if "speed" in name or "velocity" in name:
        speed_feats.append(j)
    elif "congestion" in name or "ratio" in name or "fraction" in name:
        congestion_feats.append(j)

if speed_feats:
    print(f"\nSpeed features ({len(speed_feats)}):")
    speed_oor = sum(1 for j in speed_feats if any(r[0] == j for r in out_of_range))
    print(f"  Out of range: {speed_oor}/{len(speed_feats)}")
    for j in speed_feats:
        oor = [r for r in out_of_range if r[0] == j]
        if oor:
            r = oor[0]
            print(f"  feat {j}: x_obs={r[2]:.4f} vs train=[{r[3]:.4f},{r[4]:.4f}] ({r[6]})")

if congestion_feats:
    print(f"\nCongestion features ({len(congestion_feats)}):")
    cong_oor = sum(1 for j in congestion_feats if any(r[0] == j for r in out_of_range))
    print(f"  Out of range: {cong_oor}/{len(congestion_feats)}")
    for j in congestion_feats:
        oor = [r for r in out_of_range if r[0] == j]
        if oor:
            r = oor[0]
            print(f"  feat {j}: x_obs={r[2]:.4f} vs train=[{r[3]:.4f},{r[4]:.4f}] ({r[6]})")

# ============================================================
# Key question: Is x_obs from a DIFFERENT simulation config?
# ============================================================
print("\n" + "=" * 70)
print("  KEY DIAGNOSTIC QUESTION")
print("=" * 70)

# Pattern: x_obs speed percentiles ABOVE training range
# Pattern: x_obs congestion ratios BELOW training range
# This means: x_obs has FASTER speeds and LESS congestion than any training sim

above_feats = [r for r in out_of_range if r[6] == "ABOVE"]
below_feats = [r for r in out_of_range if r[6] == "BELOW"]

print(f"\nFeatures where x_obs > training max (faster/less congested): {len(above_feats)}")
for r in above_feats:
    print(f"  feat {r[0]:2d} ({r[1]:>25}): x_obs={r[2]:.4f} > max={r[4]:.4f}")

print(f"\nFeatures where x_obs < training min (slower/more congested): {len(below_feats)}")
for r in below_feats:
    print(f"  feat {r[0]:2d} ({r[1]:>25}): x_obs={r[2]:.4f} < min={r[3]:.4f}")

# ============================================================
# DIAGNOSTIC PLOT
# ============================================================
print("\n--- Generating diagnostic plot ---")

fig, axes = plt.subplots(3, 1, figsize=(18, 14))

# 1. All features: x_obs vs training range
ax = axes[0]
n_feats = x.shape[1]
feat_indices = np.arange(n_feats)

# Training range
t_mins = x.min(axis=0)
t_maxs = x.max(axis=0)
t_means = x.mean(axis=0)

ax.fill_between(feat_indices, t_mins, t_maxs, alpha=0.3, color="steelblue",
                label="Training range")
ax.plot(feat_indices, t_means, color="steelblue", linewidth=1, alpha=0.7,
        label="Training mean")
ax.scatter(feat_indices, x_obs, color="red", s=20, zorder=5, label="x_obs")

# Highlight out-of-range
for r in out_of_range:
    ax.scatter(r[0], r[2], color="red", s=80, marker="x", zorder=6, linewidths=2)

ax.set_xlabel("Feature index")
ax.set_ylabel("Value")
ax.set_title("x_obs (red) vs Training Range (blue shaded)")
ax.legend()
ax.grid(alpha=0.3)

# 2. Z-scored view
ax = axes[1]
ax.bar(feat_indices, z_obs, color=["red" if abs(z) > 3 else "steelblue" for z in z_obs],
       alpha=0.7)
ax.axhline(3, color="red", linestyle="--", alpha=0.5, label="|z| = 3")
ax.axhline(-3, color="red", linestyle="--", alpha=0.5)
ax.set_xlabel("Feature index")
ax.set_ylabel("Z-score")
ax.set_title("x_obs Z-scores (red = |z| > 3)")
ax.legend()
ax.grid(alpha=0.3)

# 3. Distance of each training sim to x_obs
ax = axes[2]
ax.hist(distances, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
ax.axvline(closest_dist, color="red", linewidth=2, label=f"Closest: {closest_dist:.1f}")
ax.axvline(mean_dist, color="orange", linewidth=2, linestyle="--",
           label=f"Mean: {mean_dist:.1f}")
ax.set_xlabel("Euclidean distance to x_obs")
ax.set_ylabel("Count")
ax.set_title("Training sim distances to x_obs")
ax.legend()

plt.tight_layout()
plt.savefig("diagnose_mismatch.png", dpi=150)
plt.close()
print("Saved: diagnose_mismatch.png")

# ============================================================
# RECOMMENDATIONS
# ============================================================
print("\n" + "=" * 70)
print("  RECOMMENDATIONS")
print("=" * 70)

print("""
The core issue: x_obs is OUTSIDE the range of what the simulator produces
across your entire prior range. The posterior can't explain x_obs.

LIKELY CAUSES & FIXES:

1. DIFFERENT ROUTE FILE / CONFIG
   x_obs may have been generated with a different .sumocfg, route file,
   or time period than the training sims (sbi_peak.sumocfg).
   
   FIX: Regenerate x_obs using exactly the same config as training:
   
     python regenerate_xobs.py  (I will provide this)

2. DIFFERENT DEMAND LEVEL
   If x_obs has faster speeds and less congestion, it might have fewer
   vehicles than the training sims.

3. OBSERVED DATA FROM REAL WORLD (TomTom)
   If x_obs comes from TomTom/real-world data, there will always be a
   sim-to-real gap. In this case, you need:
   - Wider summary statistics that are more robust to this gap
   - Or: calibrate against a reference SUMO sim instead

4. PRIOR TOO NARROW
   Unlikely since your prior is already quite wide.

QUICK FIX for testing (use MCMC sampling):
   Modify train_snpe.py to use MCMC instead of rejection sampling.
   This will produce samples even with mismatch, but results may be
   unreliable if the mismatch is severe.
""")

# ============================================================
# HOW WAS x_obs GENERATED?
# ============================================================
print("=" * 70)
print("  CHECKING x_obs GENERATION")
print("=" * 70)
print()
print("Please check: How was x_obs.pt created?")
print()
print("If from a SUMO sim, which config files were used?")
print("If from TomTom data, that's the likely source of mismatch.")
print()

# Check if verify_matching.py or other scripts give clues
for script in ["verify_matching.py", "verify_new_stats.py", "investigate_congestion.py"]:
    if Path(script).exists():
        with open(script) as f:
            content = f.read()
        if "x_obs" in content:
            # Find the relevant lines
            lines = content.split("\n")
            print(f"\n  Found x_obs reference in {script}:")
            for i, line in enumerate(lines):
                if "x_obs" in line:
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    for l in lines[start:end]:
                        print(f"    {l}")
                    print()