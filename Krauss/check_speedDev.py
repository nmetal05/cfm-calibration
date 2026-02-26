"""
Quick check: does speedDev affect the SPREAD of speeds?
"""
import pandas as pd

df = pd.read_csv("inspect_summary.csv")

# Compare simulations that differ in speedDev
# Sim 0: default (speedDev=0.10)
# Sim 1: all_low (speedDev=0.00)
# Sim 2: all_high (speedDev=0.25)
# But these also vary other params...

# Better: compare the raw data spread
raw = pd.read_csv("inspect_raw_edgedata.csv")

print("=" * 70)
print("Effect of speedDev on speed DISTRIBUTION")
print("=" * 70)
print()

# Group by simulation label
for label in raw["sim_label"].unique():
    sim = raw[raw["sim_label"] == label]
    print(f"{label:<20s}  "
          f"sF={sim['speedFactor'].iloc[0]:.2f}  "
          f"sD={sim['speedDev'].iloc[0]:.2f}  "
          f"sig={sim['sigma'].iloc[0]:.2f}  "
          f"tau={sim['tau'].iloc[0]:.2f}  |  "
          f"mean={sim['meanSpeed'].mean():.2f}  "
          f"std={sim['meanSpeed'].std():.2f}  "
          f"p10={sim['meanSpeed'].quantile(0.1):.2f}  "
          f"p90={sim['meanSpeed'].quantile(0.9):.2f}")

print()
print("=" * 70)
print("DIRECT COMPARISON: tau effect on different metrics")
print("=" * 70)
print()

# Compare low_tau vs high_tau across ALL metrics
for label_a, label_b, param in [
    ("low_tau", "high_tau", "tau (0.5 vs 3.0)"),
    ("zero_sigma", "max_sigma", "sigma (0.0 vs 1.0)"),
    ("low_speedFactor", "high_speedFactor", "speedFactor (0.7 vs 1.2)"),
    ("all_low", "all_high", "ALL (low vs high)"),
]:
    a = raw[raw["sim_label"] == label_a]
    b = raw[raw["sim_label"] == label_b]

    print(f"\n  {param}:")
    print(f"    Mean speed:     {a['meanSpeed'].mean():.3f} vs {b['meanSpeed'].mean():.3f}  "
          f"(diff = {b['meanSpeed'].mean() - a['meanSpeed'].mean():+.3f})")
    print(f"    Std speed:      {a['meanSpeed'].std():.3f} vs {b['meanSpeed'].std():.3f}  "
          f"(diff = {b['meanSpeed'].std() - a['meanSpeed'].std():+.3f})")
    print(f"    Frac < 5 m/s:   {(a['meanSpeed']<5).mean():.4f} vs {(b['meanSpeed']<5).mean():.4f}  "
          f"(diff = {(b['meanSpeed']<5).mean() - (a['meanSpeed']<5).mean():+.4f})")
    print(f"    Waiting time:   {a['waitingTime'].sum():.0f} vs {b['waitingTime'].sum():.0f}  "
          f"(diff = {b['waitingTime'].sum() - a['waitingTime'].sum():+.0f})")
    print(f"    Time loss:      {a['timeLoss'].sum():.0f} vs {b['timeLoss'].sum():.0f}  "
          f"(diff = {b['timeLoss'].sum() - a['timeLoss'].sum():+.0f})")
    print(f"    Density mean:   {a['density'].mean():.3f} vs {b['density'].mean():.3f}  "
          f"(diff = {b['density'].mean() - a['density'].mean():+.3f})")