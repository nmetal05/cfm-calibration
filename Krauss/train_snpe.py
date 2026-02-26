"""
Train SNPE (Neural Posterior Estimation) and run inference.

Usage: python train_snpe.py
Input:  sbi_thetas.pt, sbi_xs.pt, x_obs.pt
Output: snpe_posterior.pkl, snpe_samples.pt, diagnostic plots

sbi 0.25.0 handles internal standardization — we pass raw data.
"""

import pickle
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sbi.inference import NPE
from sbi.utils import BoxUniform
from sbi.analysis import pairplot

# ============================================================
# CONFIG
# ============================================================
PARAM_NAMES = ["speedFactor", "speedDev", "sigma", "tau"]
PRIOR_LOW = torch.tensor([0.5, 0.0, 0.0, 0.5], dtype=torch.float32)
PRIOR_HIGH = torch.tensor([1.3, 0.3, 1.0, 3.0], dtype=torch.float32)
N_POSTERIOR_SAMPLES = 50_000
N_CALIB_SIMS = 200  # for SBC check

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 65)
print("  SNPE TRAINING + INFERENCE")
print("=" * 65)

theta = torch.load("sbi_thetas.pt", weights_only=True).float()
x = torch.load("sbi_xs.pt", weights_only=True).float()
x_obs = torch.load("x_obs.pt", weights_only=True).float()

print(f"\nData loaded:")
print(f"  theta:  {theta.shape}")
print(f"  x:      {x.shape}")
print(f"  x_obs:  {x_obs.shape}")

assert theta.shape == (5000, 4), f"Expected (5000,4), got {theta.shape}"
assert x.shape == (5000, 57), f"Expected (5000,57), got {x.shape}"
assert x_obs.shape == (57,), f"Expected (57,), got {x_obs.shape}"

# ============================================================
# DATA QUALITY
# ============================================================
print("\n--- Data Quality ---")

# Check for constant/degenerate columns
x_std = x.std(dim=0)
constant_mask = x_std < 1e-8
n_const = constant_mask.sum().item()
print(f"Constant columns (std < 1e-8): {n_const} / {x.shape[1]}")

if n_const > 0:
    keep_mask = ~constant_mask
    print(f"  Removing {n_const} constant columns...")
    x = x[:, keep_mask]
    x_obs = x_obs[keep_mask]
    x_std = x_std[keep_mask]
    print(f"  New shapes: x={x.shape}, x_obs={x_obs.shape}")
    torch.save(keep_mask, "snpe_feature_mask.pt")

# Check x_obs is within training range (per-feature)
x_min = x.min(dim=0).values
x_max = x.max(dim=0).values
outside = ((x_obs < x_min) | (x_obs > x_max)).sum().item()
print(f"x_obs features outside training range: {outside} / {x.shape[1]}")
if outside > 0:
    # Show which ones
    for j in range(x.shape[1]):
        if x_obs[j] < x_min[j] or x_obs[j] > x_max[j]:
            print(f"  feat {j}: x_obs={x_obs[j]:.4f}, "
                  f"train=[{x_min[j]:.4f}, {x_max[j]:.4f}]")

# Check for NaN/Inf
print(f"theta NaN: {torch.isnan(theta).any()}, Inf: {torch.isinf(theta).any()}")
print(f"x NaN:     {torch.isnan(x).any()}, Inf: {torch.isinf(x).any()}")
print(f"x_obs NaN: {torch.isnan(x_obs).any()}, Inf: {torch.isinf(x_obs).any()}")

# Parameter ranges
print(f"\nParameter distributions:")
for i, name in enumerate(PARAM_NAMES):
    col = theta[:, i]
    print(f"  {name:15s}: [{col.min():.4f}, {col.max():.4f}]  "
          f"mean={col.mean():.4f}  std={col.std():.4f}")

# Summary stat ranges
print(f"\nSummary stats: x range [{x.min():.4f}, {x.max():.4f}]")
print(f"  x_obs range: [{x_obs.min():.4f}, {x_obs.max():.4f}]")

# ============================================================
# PRIOR
# ============================================================
prior = BoxUniform(low=PRIOR_LOW, high=PRIOR_HIGH)

# ============================================================
# TRAIN NPE
# ============================================================
print("\n" + "=" * 65)
print("  TRAINING")
print("=" * 65)

# sbi 0.25.0: pass density_estimator as string
# "nsf" = Neural Spline Flows (best default for most problems)
# "maf" = Masked Autoregressive Flows (alternative)
inference = NPE(prior=prior, density_estimator="nsf")

# Append all simulations (sbi handles standardization internally)
inference.append_simulations(theta, x)

print(f"Training NPE with NSF density estimator...")
print(f"  Simulations: {len(theta)}")
print(f"  Summary stats: {x.shape[1]}")
print(f"  Parameters: {len(PARAM_NAMES)}")
print()

t_train_start = time.time()

density_estimator = inference.train(
    training_batch_size=128,
    learning_rate=5e-4,
    max_num_epochs=500,
    stop_after_epochs=30,       # early stopping patience
    clip_max_norm=5.0,
    show_train_summary=True,
)

train_time = time.time() - t_train_start
print(f"\nTraining completed in {train_time:.1f}s ({train_time/60:.1f} min)")

# Build posterior
posterior = inference.build_posterior(density_estimator)
print("Posterior built successfully")

# ============================================================
# INFERENCE ON x_obs
# ============================================================
print("\n" + "=" * 65)
print("  INFERENCE ON OBSERVED DATA")
print("=" * 65)

print(f"Sampling {N_POSTERIOR_SAMPLES} posterior samples...")
t_sample_start = time.time()
samples = posterior.sample((N_POSTERIOR_SAMPLES,), x=x_obs.unsqueeze(0))
sample_time = time.time() - t_sample_start
print(f"Sampling done in {sample_time:.1f}s")

samples_np = samples.numpy()

# Point estimates
print(f"\n{'Parameter':>15} | {'Mean':>8} | {'Median':>8} | {'Std':>8} | "
      f"{'5%':>8} | {'95%':>8} | {'MAP est':>8}")
print("-" * 85)

map_estimates = []
for i, name in enumerate(PARAM_NAMES):
    col = samples_np[:, i]
    mean = col.mean()
    median = np.median(col)
    std = col.std()
    q05, q95 = np.percentile(col, [5, 95])

    # MAP via KDE
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(col)
        grid = np.linspace(col.min(), col.max(), 1000)
        map_val = grid[kde(grid).argmax()]
    except Exception:
        map_val = median
    map_estimates.append(map_val)

    print(f"{name:>15} | {mean:8.4f} | {median:8.4f} | {std:8.4f} | "
          f"{q05:8.4f} | {q95:8.4f} | {map_val:8.4f}")

# ============================================================
# DIAGNOSTICS
# ============================================================
print("\n" + "=" * 65)
print("  DIAGNOSTICS")
print("=" * 65)

# --- 1. Pairplot ---
print("\n[1/4] Generating pairplot...")
fig, axes = pairplot(
    samples_np,
    labels=PARAM_NAMES,
    limits=list(zip(PRIOR_LOW.numpy(), PRIOR_HIGH.numpy())),
    figsize=(10, 10),
    points_offdiag={"markersize": 5},
    points_colors=["red"],
)
fig.suptitle("SNPE Posterior from x_obs", fontsize=14, y=1.02)
plt.savefig("snpe_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: snpe_pairplot.png")

# --- 2. Marginal posteriors with prior ---
print("[2/4] Generating marginal plots...")
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for i, (name, ax) in enumerate(zip(PARAM_NAMES, axes)):
    col = samples_np[:, i]
    lo, hi = PRIOR_LOW[i].item(), PRIOR_HIGH[i].item()

    # Posterior histogram
    ax.hist(col, bins=80, density=True, alpha=0.7, color="steelblue",
            edgecolor="white", linewidth=0.3, label="Posterior")

    # Prior (uniform)
    ax.axhline(1.0 / (hi - lo), color="gray", linestyle="--",
               alpha=0.6, label="Prior")

    # Point estimates
    ax.axvline(col.mean(), color="red", linewidth=1.5, label=f"Mean={col.mean():.3f}")
    q05, q95 = np.percentile(col, [5, 95])
    ax.axvspan(q05, q95, alpha=0.15, color="red", label=f"90% CI")

    ax.set_xlabel(name, fontsize=12)
    ax.set_ylabel("Density" if i == 0 else "")
    ax.set_xlim(lo, hi)
    ax.legend(fontsize=8)
    ax.set_title(name)

plt.suptitle("SNPE Marginal Posteriors (blue) vs Prior (gray dashed)", fontsize=13)
plt.tight_layout()
plt.savefig("snpe_marginals.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: snpe_marginals.png")

# --- 3. Posterior recovery / SBC check ---
print(f"[3/4] Simulation-based calibration ({N_CALIB_SIMS} sims)...")
np.random.seed(42)
calib_indices = np.random.choice(len(theta), N_CALIB_SIMS, replace=False)

ci_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
coverage_by_level = {level: [] for level in ci_levels}
recovery_errors = []

for k, idx in enumerate(calib_indices):
    x_test = x[idx].unsqueeze(0)
    true_theta = theta[idx].numpy()

    try:
        samps = posterior.sample((2000,), x=x_test).numpy()
    except Exception as e:
        print(f"  Warning: sampling failed for idx {idx}: {e}")
        continue

    for i in range(4):
        error = samps[:, i].mean() - true_theta[i]
        recovery_errors.append({"param": i, "error": error})

        for level in ci_levels:
            alpha = (1 - level) / 2
            q_lo = np.percentile(samps[:, i], alpha * 100)
            q_hi = np.percentile(samps[:, i], (1 - alpha) * 100)
            coverage_by_level[level].append(q_lo <= true_theta[i] <= q_hi)

    if (k + 1) % 50 == 0:
        print(f"    {k+1}/{N_CALIB_SIMS} done...")

# Coverage results
print(f"\n  Calibration results:")
expected_cov = []
observed_cov = []
for level in ci_levels:
    obs = np.mean(coverage_by_level[level])
    expected_cov.append(level)
    observed_cov.append(obs)
    status = "OK" if abs(obs - level) < 0.1 else "WARN"
    print(f"    {level*100:4.0f}% CI: observed {obs*100:5.1f}%  [{status}]")

# Calibration plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Ideal", linewidth=1.5)
ax.plot(expected_cov, observed_cov, "o-", color="steelblue",
        markersize=8, linewidth=2, label="SNPE")
ax.fill_between([0, 1], [0-0.05, 1-0.05], [0+0.05, 1+0.05],
                alpha=0.1, color="gray", label="±5% band")
ax.set_xlabel("Expected coverage", fontsize=12)
ax.set_ylabel("Observed coverage", fontsize=12)
ax.set_title("Simulation-Based Calibration", fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(0.45, 1.0)
ax.set_ylim(0.45, 1.0)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("snpe_calibration.png", dpi=150)
plt.close()
print("  Saved: snpe_calibration.png")

# --- 4. Per-parameter recovery bias ---
print("[4/4] Parameter recovery bias...")
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for i, (name, ax) in enumerate(zip(PARAM_NAMES, axes)):
    errors = [r["error"] for r in recovery_errors if r["param"] == i]
    ax.hist(errors, bins=40, alpha=0.7, color="coral", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
    mean_err = np.mean(errors)
    ax.axvline(mean_err, color="red", linewidth=1.5,
               label=f"Mean bias={mean_err:.4f}")
    ax.set_xlabel(f"Posterior mean - True ({name})")
    ax.set_ylabel("Count" if i == 0 else "")
    ax.legend(fontsize=8)
    ax.set_title(name)

plt.suptitle("Posterior Recovery Bias (should be centered at 0)", fontsize=13)
plt.tight_layout()
plt.savefig("snpe_recovery_bias.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: snpe_recovery_bias.png")

# ============================================================
# SAVE EVERYTHING
# ============================================================
print("\n" + "=" * 65)
print("  SAVING")
print("=" * 65)

# Posterior
with open("snpe_posterior.pkl", "wb") as f:
    pickle.dump(posterior, f)
print("  snpe_posterior.pkl")

# Samples
torch.save(samples, "snpe_samples.pt")
print("  snpe_samples.pt")

# Point estimates
results = {
    "param_names": PARAM_NAMES,
    "map": np.array(map_estimates),
    "mean": samples_np.mean(axis=0),
    "median": np.median(samples_np, axis=0),
    "std": samples_np.std(axis=0),
    "q05": np.percentile(samples_np, 5, axis=0),
    "q95": np.percentile(samples_np, 95, axis=0),
    "calibration_expected": expected_cov,
    "calibration_observed": observed_cov,
    "train_time_s": train_time,
    "n_sims": len(theta),
}
np.savez("snpe_results.npz", **results)
print("  snpe_results.npz")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 65)
print("  RESULTS SUMMARY")
print("=" * 65)
print(f"\nTraining:  {len(theta)} sims, {train_time:.0f}s")
print(f"Sampling:  {N_POSTERIOR_SAMPLES} posterior samples, {sample_time:.1f}s")
print(f"\nInferred parameters (from x_obs):")
print(f"{'Parameter':>15} | {'MAP':>8} | {'Mean±Std':>15} | {'90% CI':>18}")
print("-" * 65)
for i, name in enumerate(PARAM_NAMES):
    col = samples_np[:, i]
    q05, q95 = np.percentile(col, [5, 95])
    print(f"{name:>15} | {map_estimates[i]:8.4f} | "
          f"{col.mean():7.4f} ± {col.std():.4f} | "
          f"[{q05:.4f}, {q95:.4f}]")

print(f"\nCalibration: ", end="")
max_deviation = max(abs(e - o) for e, o in zip(expected_cov, observed_cov))
if max_deviation < 0.05:
    print(f"EXCELLENT (max deviation {max_deviation:.1%})")
elif max_deviation < 0.10:
    print(f"GOOD (max deviation {max_deviation:.1%})")
elif max_deviation < 0.15:
    print(f"ACCEPTABLE (max deviation {max_deviation:.1%})")
else:
    print(f"POOR (max deviation {max_deviation:.1%}) — consider more sims or different architecture")

print(f"\nFiles:")
print(f"  snpe_posterior.pkl       — trained posterior object")
print(f"  snpe_samples.pt         — {N_POSTERIOR_SAMPLES} posterior samples")
print(f"  snpe_results.npz        — point estimates + calibration")
print(f"  snpe_pairplot.png       — joint posterior")
print(f"  snpe_marginals.png      — marginal posteriors vs prior")
print(f"  snpe_calibration.png    — SBC calibration curve")
print(f"  snpe_recovery_bias.png  — parameter recovery bias")