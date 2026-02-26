"""Verify the new summary statistics capture all parameter effects"""
from parse_output import parse_edge_data, summary_statistics
from pathlib import Path

# Use the inspect_runs data we already generated
labels = [
    "default", "all_low", "all_high",
    "low_speedFactor", "high_speedFactor",
    "zero_sigma", "max_sigma",
    "low_tau", "high_tau",
    "random_mid"
]

results = {}
for i, label in enumerate(labels):
    edgedata_path = Path(f"inspect_runs/sim_{i:02d}/edgedata.xml")
    if edgedata_path.exists():
        data = parse_edge_data(str(edgedata_path))
        x = summary_statistics(data)
        results[label] = x
        if i == 0:
            print(f"Summary stats length: {len(x)}")

print()
print("=" * 60)
print("DISTANCE BETWEEN SIMULATIONS (L2 norm of summary stats)")
print("=" * 60)
print()

import numpy as np

# Compare each to default
default = results["default"]
print(f"{'Comparison':<35s} {'L2 distance':>12s} {'Relative':>10s}")
print("-" * 60)

for label in labels:
    if label == "default":
        continue
    dist = np.linalg.norm(results[label] - default)
    rel = dist / np.linalg.norm(default)
    print(f"default vs {label:<23s} {dist:12.2f} {rel:10.3f}")

print()
print("=" * 60)
print("KEY COMPARISONS (should all show big differences now)")
print("=" * 60)
print()

# tau: should now show bigger difference with density/waiting time included
dist_tau = np.linalg.norm(results["low_tau"] - results["high_tau"])
dist_sf = np.linalg.norm(results["low_speedFactor"] - results["high_speedFactor"])
dist_sig = np.linalg.norm(results["zero_sigma"] - results["max_sigma"])

print(f"speedFactor (0.7 vs 1.2):  L2 = {dist_sf:.2f}")
print(f"sigma       (0.0 vs 1.0):  L2 = {dist_sig:.2f}")
print(f"tau         (0.5 vs 3.0):  L2 = {dist_tau:.2f}")
print()

if dist_tau > 5.0:
    print("✅ tau IS distinguishable with enriched stats → KEEP all 4 params")
    print()
    print("FINAL SBI CONFIGURATION:")
    print("  Parameters: [speedFactor, speedDev, sigma, tau]")
    print("  Prior:      [0.7-1.2, 0.0-0.25, 0.0-1.0, 0.5-3.0]")
    print(f"  Summary stats dimension: {len(default)}")
    print("  Recommended sims: 5000")
    print(f"  Estimated time: ~{5000/2500:.1f} hours")
else:
    print("⚠️  tau still hard to distinguish → consider dropping it")
    print()
    print("FINAL SBI CONFIGURATION:")
    print("  Parameters: [speedFactor, speedDev, sigma]")
    print("  Prior:      [0.7-1.2, 0.0-0.25, 0.0-1.0]")
    print(f"  Summary stats dimension: {len(default)}")
    print("  Recommended sims: 3000")
    print(f"  Estimated time: ~{3000/2500:.1f} hours")