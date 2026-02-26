import numpy as np
from pathlib import Path

krauss_dir = Path(r"C:\Users\PC\Downloads\Krauss")

snpe = np.load(krauss_dir / "snpe_results.npz", allow_pickle=True)
asnpe = np.load(krauss_dir / "asnpe_results.npz", allow_pickle=True)

print("=" * 60)
print("SNPE vs ASNPE Comparison for Krauss")
print("=" * 60)

print("\n--- Mean Estimates ---")
print(f"{'Param':<15} {'SNPE':>12} {'ASNPE':>12} {'Diff':>10}")
print("-" * 50)
for i, name in enumerate(snpe["param_names"]):
    diff = snpe["mean"][i] - asnpe["mean"][i]
    print(
        f"{name:<15} {snpe['mean'][i]:>12.4f} {asnpe['mean'][i]:>12.4f} {diff:>+10.4f}"
    )

print("\n--- Standard Deviation (uncertainty) ---")
print(f"{'Param':<15} {'SNPE':>12} {'ASNPE':>12} {'Better':>10}")
print("-" * 50)
snpe_wins = 0
asnpe_wins = 0
for i, name in enumerate(snpe["param_names"]):
    if snpe["std"][i] < asnpe["std"][i]:
        winner = "SNPE"
        snpe_wins += 1
    else:
        winner = "ASNPE"
        asnpe_wins += 1
    print(f"{name:<15} {snpe['std'][i]:>12.4f} {asnpe['std'][i]:>12.4f} {winner:>10}")

print(f"\n--- Summary ---")
print(f"SNPE wins: {snpe_wins}/4")
print(f"ASNPE wins: {asnpe_wins}/4")

# Check total simulation counts
print(f"\n--- Simulation Budget ---")
print(f"SNPE total sims: {snpe.get('total_sims', 'N/A')}")
print(f"ASNPE total sims: {asnpe.get('total_sims', 'N/A')}")
