"""
Compare Car-Following Models: Krauss, Wiedemann, IDM
Loads SNPE results from each model folder and compares them
"""

import numpy as np
from pathlib import Path

# Define paths to each model's results
MODEL_PATHS = {
    "Krauss": Path(r"C:\Users\PC\Downloads\Krauss"),
    "Wiedemann": Path(r"C:\Users\PC\Downloads\Wiedemann\Wiedemann"),
    "IDM": Path(r"C:\Users\PC\Downloads\IDM\IDM"),
}

# Load results
results = {}
for model_name, model_path in MODEL_PATHS.items():
    results[model_name] = np.load(model_path / "snpe_results.npz", allow_pickle=True)
    print(f"Loaded {model_name} from {model_path.name}")

print("\n" + "=" * 80)
print(" " * 25 + "CAR-FOLLOWING MODEL COMPARISON")
print("=" * 80)

# Get parameter names (should be the same for all)
param_names = results["Krauss"]["param_names"]
print(f"\nParameters: {list(param_names)}")

# ============================================================================
# TABLE 1: Mean Estimates
# ============================================================================
print("\n" + "=" * 80)
print("TABLE 1: Mean Parameter Estimates")
print("=" * 80)
print(f"\n{'Parameter':<15} | {'Krauss':>10} | {'Wiedemann':>10} | {'IDM':>10}")
print("-" * 55)
for i, param in enumerate(param_names):
    krauss = results["Krauss"]["mean"][i]
    wiedemann = results["Wiedemann"]["mean"][i]
    idm = results["IDM"]["mean"][i]
    print(f"{param:<15} | {krauss:>10.4f} | {wiedemann:>10.4f} | {idm:>10.4f}")

# ============================================================================
# TABLE 2: Standard Deviation (Uncertainty)
# ============================================================================
print("\n" + "=" * 80)
print("TABLE 2: Standard Deviation (Uncertainty - lower is better)")
print("=" * 80)
print(
    f"\n{'Parameter':<15} | {'Krauss':>10} | {'Wiedemann':>10} | {'IDM':>10} | {'Best':>12}"
)
print("-" * 70)
for i, param in enumerate(param_names):
    krauss = results["Krauss"]["std"][i]
    wiedemann = results["Wiedemann"]["std"][i]
    idm = results["IDM"]["std"][i]

    # Find best (lowest std)
    stds = {"Krauss": krauss, "Wiedemann": wiedemann, "IDM": idm}
    best = min(stds, key=stds.get)

    print(
        f"{param:<15} | {krauss:>10.4f} | {wiedemann:>10.4f} | {idm:>10.4f} | {best:>12}"
    )

# ============================================================================
# TABLE 3: 90% Credible Intervals
# ============================================================================
print("\n" + "=" * 80)
print("TABLE 3: 90% Credible Intervals")
print("=" * 80)
print(f"\n{'Parameter':<15} | {'Krauss':>20} | {'Wiedemann':>20} | {'IDM':>20}")
print("-" * 80)
for i, param in enumerate(param_names):
    k_q05 = results["Krauss"]["q05"][i]
    k_q95 = results["Krauss"]["q95"][i]
    w_q05 = results["Wiedemann"]["q05"][i]
    w_q95 = results["Wiedemann"]["q95"][i]
    i_q05 = results["IDM"]["q05"][i]
    i_q95 = results["IDM"]["q95"][i]

    print(
        f"{param:<15} | [{k_q05:.3f}, {k_q95:.3f}]{'':<5} | [{w_q05:.3f}, {w_q95:.3f}]{'':<5} | [{i_q05:.3f}, {i_q95:.3f}]"
    )

# ============================================================================
# TABLE 4: CI Width (Uncertainty measure)
# ============================================================================
print("\n" + "=" * 80)
print("TABLE 4: 90% CI Width (higher = more uncertainty)")
print("=" * 80)
print(
    f"\n{'Parameter':<15} | {'Krauss':>10} | {'Wiedemann':>10} | {'IDM':>10} | {'Best':>12}"
)
print("-" * 70)
for i, param in enumerate(param_names):
    krauss = results["Krauss"]["q95"][i] - results["Krauss"]["q05"][i]
    wiedemann = results["Wiedemann"]["q95"][i] - results["Wiedemann"]["q05"][i]
    idm = results["IDM"]["q95"][i] - results["IDM"]["q05"][i]

    # Find best (lowest width)
    widths = {"Krauss": krauss, "Wiedemann": wiedemann, "IDM": idm}
    best = min(widths, key=widths.get)

    print(
        f"{param:<15} | {krauss:>10.3f} | {wiedemann:>10.3f} | {idm:>10.3f} | {best:>12}"
    )

# ============================================================================
# TABLE 5: MAP Estimates
# ============================================================================
print("\n" + "=" * 80)
print("TABLE 5: Maximum A Posteriori (MAP) Estimates")
print("=" * 80)
print(f"\n{'Parameter':<15} | {'Krauss':>10} | {'Wiedemann':>10} | {'IDM':>10}")
print("-" * 55)
for i, param in enumerate(param_names):
    krauss = results["Krauss"]["map"][i]
    wiedemann = results["Wiedemann"]["map"][i]
    idm = results["IDM"]["map"][i]
    print(f"{param:<15} | {krauss:>10.4f} | {wiedemann:>10.4f} | {idm:>10.4f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print(" " * 30 + "SUMMARY")
print("=" * 80)

print("\n--- Calibration Quality (lower is better) ---")
try:
    for model in ["Krauss", "Wiedemann", "IDM"]:
        cal_exp = results[model].get("calibration_expected", [])
        cal_obs = results[model].get("calibration_observed", [])
        if len(cal_exp) > 0:
            max_dev = max(abs(e - o) for e, o in zip(cal_exp, cal_obs))
            print(f"  {model}: max deviation = {max_dev * 100:.1f}%")
except:
    print("  (Calibration data not available)")

print("\n--- Best Model by Uncertainty (CI Width) ---")
for i, param in enumerate(param_names):
    krauss = results["Krauss"]["q95"][i] - results["Krauss"]["q05"][i]
    wiedemann = results["Wiedemann"]["q95"][i] - results["Wiedemann"]["q05"][i]
    idm = results["IDM"]["q95"][i] - results["IDM"]["q05"][i]

    widths = {"Krauss": krauss, "Wiedemann": wiedemann, "IDM": idm}
    best = min(widths, key=widths.get)
    print(f"  {param}: {best} (width={widths[best]:.3f})")

print("\n--- Physical Interpretation ---")
print("\n  speedFactor:")
print(
    f"    - Krauss: {results['Krauss']['map'][0]:.2f} (drivers go {(results['Krauss']['map'][0] - 1) * 100:+.0f}% of speed limit)"
)
print(
    f"    - Wiedemann: {results['Wiedemann']['map'][0]:.2f} (drivers go {(results['Wiedemann']['map'][0] - 1) * 100:+.0f}% of speed limit)"
)
print(
    f"    - IDM: {results['IDM']['map'][0]:.2f} (drivers go {(results['IDM']['map'][0] - 1) * 100:+.0f}% of speed limit)"
)

print("\n  speedDev:")
print(f"    - Krauss: {results['Krauss']['map'][1]:.3f}")
print(f"    - Wiedemann: {results['Wiedemann']['map'][1]:.3f}")
print(f"    - IDM: {results['IDM']['map'][1]:.3f}")

print("\n  sigma (driver imperfection):")
print(f"    - Krauss: {results['Krauss']['map'][2]:.2f} (high = more random/imperfect)")
print(f"    - Wiedemann: {results['Wiedemann']['map'][2]:.2f}")
print(f"    - IDM: {results['IDM']['map'][2]:.2f}")

print("\n  tau (time headway in seconds):")
print(f"    - Krauss: {results['Krauss']['map'][3]:.2f}s")
print(f"    - Wiedemann: {results['Wiedemann']['map'][3]:.2f}s")
print(f"    - IDM: {results['IDM']['map'][3]:.2f}s")

print("\n" + "=" * 80)
print(" " * 35 + "END")
print("=" * 80)
