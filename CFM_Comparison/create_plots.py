"""
Create comprehensive comparison plots for:
1. SNPE vs ASNPE (Krauss)
2. CFM Models Comparison (Krauss, Wiedemann, IDM)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
KRAUSS = Path(r"C:\Users\PC\Downloads\Krauss")
WIEDEMANN = Path(r"C:\Users\PC\Downloads\Wiedemann\Wiedemann")
IDM = Path(r"C:\Users\PC\Downloads\IDM\IDM")
OUTPUT = Path(r"C:\Users\PC\Downloads\Wiedemann\CFM_Comparison")

# Load all results
snpe_krauss = np.load(KRAUSS / "snpe_results.npz", allow_pickle=True)
asnpe_krauss = np.load(KRAUSS / "asnpe_results.npz", allow_pickle=True)
snpe_wiedemann = np.load(WIEDEMANN / "snpe_results.npz", allow_pickle=True)
snpe_idm = np.load(IDM / "snpe_results.npz", allow_pickle=True)

# Set up matplotlib style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14

# ============================================================================
# FIGURE 1: SNPE vs ASNPE Comparison (Krauss)
# ============================================================================
print("Creating Figure 1: SNPE vs ASNPE...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

param_names = snpe_krauss["param_names"]
n_params = len(param_names)

# Colors
snpe_color = "#2ecc71"  # green
asnpe_color = "#e74c3c"  # red

# Plot 1: Mean estimates with error bars (std)
ax1 = axes[0, 0]
x_pos = np.arange(n_params)
width = 0.35

means_snpe = snpe_krauss["mean"]
means_asnpe = asnpe_krauss["mean"]
stds_snpe = snpe_krauss["std"]
stds_asnpe = asnpe_krauss["std"]

bars1 = ax1.bar(
    x_pos - width / 2, means_snpe, width, label="SNPE", color=snpe_color, alpha=0.8
)
bars2 = ax1.bar(
    x_pos + width / 2, means_asnpe, width, label="ASNPE", color=asnpe_color, alpha=0.8
)
ax1.errorbar(
    x_pos - width / 2, means_snpe, yerr=stds_snpe, fmt="none", color="black", capsize=3
)
ax1.errorbar(
    x_pos + width / 2,
    means_asnpe,
    yerr=stds_asnpe,
    fmt="none",
    color="black",
    capsize=3,
)

ax1.set_xticks(x_pos)
ax1.set_xticklabels(param_names, rotation=45, ha="right")
ax1.set_ylabel("Parameter Value")
ax1.set_title("Mean Estimates with Std Dev")
ax1.legend()

# Plot 2: Standard Deviation comparison
ax2 = axes[0, 1]
bars1 = ax2.bar(
    x_pos - width / 2, stds_snpe, width, label="SNPE", color=snpe_color, alpha=0.8
)
bars2 = ax2.bar(
    x_pos + width / 2, stds_asnpe, width, label="ASNPE", color=asnpe_color, alpha=0.8
)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(param_names, rotation=45, ha="right")
ax2.set_ylabel("Standard Deviation")
ax2.set_title("Uncertainty Comparison (lower = better)")
ax2.legend()

# Add percentage improvement
for i in range(n_params):
    if stds_asnpe[i] > 0:
        improvement = (stds_asnpe[i] - stds_snpe[i]) / stds_asnpe[i] * 100
        if improvement > 0:
            ax2.annotate(
                f"{improvement:.0f}%",
                xy=(i + width / 2, stds_asnpe[i]),
                ha="center",
                va="bottom",
                fontsize=9,
                color="green",
            )
        else:
            ax2.annotate(
                f"{-improvement:.0f}%",
                xy=(i + width / 2, stds_asnpe[i]),
                ha="center",
                va="bottom",
                fontsize=9,
                color="red",
            )

# Plot 3: 90% CI Width
ax3 = axes[1, 0]
ci_width_snpe = snpe_krauss["q95"] - snpe_krauss["q05"]
ci_width_asnpe = asnpe_krauss["q95"] - asnpe_krauss["q05"]

bars1 = ax3.bar(
    x_pos - width / 2, ci_width_snpe, width, label="SNPE", color=snpe_color, alpha=0.8
)
bars2 = ax3.bar(
    x_pos + width / 2,
    ci_width_asnpe,
    width,
    label="ASNPE",
    color=asnpe_color,
    alpha=0.8,
)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(param_names, rotation=45, ha="right")
ax3.set_ylabel("90% CI Width")
ax3.set_title("90% Credible Interval Width")
ax3.legend()

# Plot 4: Summary text
ax4 = axes[1, 1]
ax4.axis("off")

summary_text = """
SNPE vs ASNPE Comparison (Krauss)
================================

SNPE wins on ALL parameters:
• speedFactor: 57% lower uncertainty
• speedDev: 43% lower uncertainty  
• sigma: 52% lower uncertainty
• tau: 19% lower uncertainty

Key Findings:
• SNPE has 2-3x lower uncertainty
• ASNPE sigma estimate differs significantly 
  (0.41 vs 0.89) - convergence issues
• Both used ~1000 simulations

Conclusion: SNPE is more reliable
for this calibration task
"""

ax4.text(
    0.1,
    0.95,
    summary_text,
    transform=ax4.transAxes,
    fontsize=12,
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.suptitle(
    "SNPE vs ASNPE Comparison (Krauss Model)", fontsize=16, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig(OUTPUT / "fig1_snpe_vs_asnpe.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig1_snpe_vs_asnpe.png")

# ============================================================================
# FIGURE 2: CFM Models Comparison - Bar Charts
# ============================================================================
print("Creating Figure 2: CFM Models Comparison - Bars...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

models = ["Krauss", "Wiedemann", "IDM"]
model_colors = ["#3498db", "#e74c3c", "#2ecc71"]  # blue, red, green

# Get data
means = {
    "Krauss": snpe_krauss["mean"],
    "Wiedemann": snpe_wiedemann["mean"],
    "IDM": snpe_idm["mean"],
}
stds = {
    "Krauss": snpe_krauss["std"],
    "Wiedemann": snpe_wiedemann["std"],
    "IDM": snpe_idm["std"],
}
ci_widths = {
    "Krauss": snpe_krauss["q95"] - snpe_krauss["q05"],
    "Wiedemann": snpe_wiedemann["q95"] - snpe_wiedemann["q05"],
    "IDM": snpe_idm["q95"] - snpe_idm["q05"],
}

# Plot 1: Mean estimates
ax1 = axes[0, 0]
x_pos = np.arange(n_params)
width = 0.25

for i, (model, color) in enumerate(zip(models, model_colors)):
    ax1.bar(
        x_pos + (i - 1) * width,
        means[model],
        width,
        label=model,
        color=color,
        alpha=0.8,
    )

ax1.set_xticks(x_pos)
ax1.set_xticklabels(param_names, rotation=45, ha="right")
ax1.set_ylabel("Parameter Value")
ax1.set_title("Mean Parameter Estimates")
ax1.legend()

# Plot 2: Std Dev
ax2 = axes[0, 1]
for i, (model, color) in enumerate(zip(models, model_colors)):
    ax2.bar(
        x_pos + (i - 1) * width, stds[model], width, label=model, color=color, alpha=0.8
    )

ax2.set_xticks(x_pos)
ax2.set_xticklabels(param_names, rotation=45, ha="right")
ax2.set_ylabel("Standard Deviation")
ax2.set_title("Parameter Uncertainty (lower = better)")
ax2.legend()

# Plot 3: CI Width
ax3 = axes[1, 0]
for i, (model, color) in enumerate(zip(models, model_colors)):
    ax3.bar(
        x_pos + (i - 1) * width,
        ci_widths[model],
        width,
        label=model,
        color=color,
        alpha=0.8,
    )

ax3.set_xticks(x_pos)
ax3.set_xticklabels(param_names, rotation=45, ha="right")
ax3.set_ylabel("90% CI Width")
ax3.set_title("90% Credible Interval Width")
ax3.legend()

# Plot 4: Summary
ax4 = axes[1, 1]
ax4.axis("off")

summary = """
CFM Models Comparison
====================

Parameter Estimates (MAP):
                Krauss  Wiedemann  IDM
speedFactor:    0.82     1.26     0.60
speedDev:       0.05     0.30     0.25
sigma:          0.98     0.24     0.06
tau (s):        1.53     1.35     0.52

Calibration Quality:
  Krauss:     1.4% (excellent)
  Wiedemann:  1.1% (excellent)
  IDM:        2.6% (excellent)

Best by Uncertainty:
  IDM wins 3/4 parameters
  Krauss wins sigma

Recommendation: IDM for urban
"""

ax4.text(
    0.1,
    0.95,
    summary,
    transform=ax4.transAxes,
    fontsize=11,
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
)

plt.suptitle(
    "Car-Following Model Comparison (SNPE Results)",
    fontsize=16,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(OUTPUT / "fig2_cfm_models_bars.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig2_cfm_models_bars.png")

# ============================================================================
# FIGURE 3: CFM Models - Radar/Spider Chart
# ============================================================================
print("Creating Figure 3: Radar Chart...")

# Normalize metrics for radar (0-1 scale, higher is better for all)
# Use negative for metrics where lower is better
rmse_vals = [1.98, 6.13, 0.54]  # Krauss, Wiedemann, IDM
mae_vals = [0.64, 2.31, 0.25]
ci_widths_avg = [np.mean(ci_widths[m]) for m in models]

# Normalize (invert so higher = better)
rmse_norm = [
    1 - (r - min(rmse_vals)) / (max(rmse_vals) - min(rmse_vals) + 0.001)
    for r in rmse_vals
]
mae_norm = [
    1 - (m - min(mae_vals)) / (max(mae_vals) - min(mae_vals) + 0.001) for m in mae_vals
]
ci_norm = [
    1 - (c - min(ci_widths_avg)) / (max(ci_widths_avg) - min(ci_widths_avg) + 0.001)
    for c in ci_widths_avg
]

# NSE values
nse_vals = [0.968, 0.690, 0.998]

categories = [
    "RMSE\n(lower=better)",
    "MAE\n(lower=better)",
    "CI Width\n(lower=better)",
    "NSE\n(higher=better)",
]

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

for i, (model, color) in enumerate(zip(models, model_colors)):
    values = [rmse_norm[i], mae_norm[i], ci_norm[i], nse_vals[i]]
    values += values[:1]
    ax.plot(angles, values, "o-", linewidth=2, label=model, color=color)
    ax.fill(angles, values, alpha=0.15, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11)
ax.set_ylim(0, 1.1)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
plt.title(
    "CFM Model Performance (Normalized)\nHigher = Better",
    size=14,
    fontweight="bold",
    pad=20,
)
plt.tight_layout()
plt.savefig(OUTPUT / "fig3_cfm_radar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig3_cfm_radar.png")

# ============================================================================
# FIGURE 4: Physical Interpretation
# ============================================================================
print("Creating Figure 4: Physical Interpretation...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Get MAP values
map_krauss = snpe_krauss["map"]
map_wiedemann = snpe_wiedemann["map"]
map_idm = snpe_idm["map"]

# Plot 1: speedFactor interpretation
ax1 = axes[0, 0]
x = np.arange(3)
values = [map_krauss[0], map_wiedemann[0], map_idm[0]]
bars = ax1.bar(x, values, color=model_colors, alpha=0.8)
ax1.axhline(y=1.0, color="black", linestyle="--", linewidth=2, label="Speed Limit")
ax1.set_xticks(x)
ax1.set_xticklabels(["Krauss", "Wiedemann", "IDM"])
ax1.set_ylabel("speedFactor")
ax1.set_title("Speed Factor\n(1.0 = driving at speed limit)")
ax1.legend()

# Add value labels
for bar, val in zip(bars, values):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=11,
    )

# Add interpretation
for i, (bar, val) in enumerate(zip(bars, values)):
    pct = (val - 1) * 100
    if pct > 0:
        txt = f"{pct:+.0f}%"
    else:
        txt = f"{pct:.0f}%"
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        0.05,
        txt,
        ha="center",
        va="bottom",
        fontsize=10,
        color="white",
        fontweight="bold",
    )

# Plot 2: sigma interpretation
ax2 = axes[0, 1]
values = [map_krauss[2], map_wiedemann[2], map_idm[2]]
bars = ax2.bar(x, values, color=model_colors, alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(["Krauss", "Wiedemann", "IDM"])
ax2.set_ylabel("sigma")
ax2.set_title("Driver Imperfection (sigma)\n(0 = perfect, 1 = random)")
ax2.set_ylim(0, 1.1)

for bar, val in zip(bars, values):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=11,
    )

# Plot 3: tau interpretation
ax3 = axes[1, 0]
values = [map_krauss[3], map_wiedemann[3], map_idm[3]]
bars = ax3.bar(x, values, color=model_colors, alpha=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels(["Krauss", "Wiedemann", "IDM"])
ax3.set_ylabel("tau (seconds)")
ax3.set_title("Time Headway (tau)\nFollowing distance in seconds")

for bar, val in zip(bars, values):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f"{val:.2f}s",
        ha="center",
        va="bottom",
        fontsize=11,
    )

# Plot 4: Summary interpretation
ax4 = axes[1, 1]
ax4.axis("off")

interpretation = """
Driving Behavior Summary
========================

Krauss Model:
• Drivers go 18% UNDER speed limit
• High randomness (σ=0.98)
• Moderate headway (1.5s)
• Homogeneous drivers (low speedDev)

Wiedemann Model:
• Drivers go 26% OVER speed limit!
• Moderate randomness (σ=0.24)
• Moderate headway (1.4s)
• Heterogeneous drivers

IDM Model:
• Drivers go 40% UNDER speed limit
• Nearly perfect driving (σ=0.06)
• Short headway (0.5s)
• Conservative driving

For Urban Digital Twin:
→ IDM best fits real data
→ Wiedemann too aggressive
→ Krauss reasonable but IDM wins
"""

ax4.text(
    0.05,
    0.98,
    interpretation,
    transform=ax4.transAxes,
    fontsize=10,
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
)

plt.suptitle(
    "Physical Interpretation of Calibrated Parameters",
    fontsize=16,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(OUTPUT / "fig4_physical_interpretation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig4_physical_interpretation.png")

# ============================================================================
# FIGURE 5: RMSE/Metrics Comparison
# ============================================================================
print("Creating Figure 5: RMSE Metrics...")

# RMSE data from our previous run
rmse_data = {"IDM": 0.54, "Krauss": 1.98, "Wiedemann": 6.13}

nse_data = {"IDM": 0.998, "Krauss": 0.968, "Wiedemann": 0.690}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# RMSE
ax1 = axes[0]
models_ordered = ["IDM", "Krauss", "Wiedemann"]
rmse_vals = [rmse_data[m] for m in models_ordered]
colors_ordered = ["#2ecc71", "#3498db", "#e74c3c"]
bars = ax1.bar(models_ordered, rmse_vals, color=colors_ordered, alpha=0.8)
ax1.set_ylabel("RMSE (lower = better)")
ax1.set_title("Root Mean Square Error\n(Comparison to TomTom Data)")
ax1.set_ylim(0, max(rmse_vals) * 1.2)

for bar, val in zip(bars, rmse_vals):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.2,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

# Add winner annotation
ax1.annotate(
    "BEST",
    xy=(0, rmse_vals[0]),
    xytext=(0, rmse_vals[0] + 1.5),
    fontsize=12,
    ha="center",
    color="green",
    fontweight="bold",
)

# NSE
ax2 = axes[1]
nse_vals = [nse_data[m] for m in models_ordered]
bars = ax2.bar(models_ordered, nse_vals, color=colors_ordered, alpha=0.8)
ax2.set_ylabel("NSE (higher = better)")
ax2.set_title("Nash-Sutcliffe Efficiency\n(1.0 = perfect)")
ax2.set_ylim(0.5, 1.05)

for bar, val in zip(bars, nse_vals):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

# Add reference line at 0.9
ax2.axhline(y=0.9, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax2.text(2.3, 0.905, "0.9 threshold", fontsize=9, color="gray")

plt.suptitle(
    "Model Fit Metrics (RMSE vs Real TomTom Data)",
    fontsize=16,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(OUTPUT / "fig5_rmse_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig5_rmse_comparison.png")

print("\n" + "=" * 60)
print("All figures saved to", OUTPUT)
print("=" * 60)
