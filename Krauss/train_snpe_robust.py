"""
Train SNPE with robust feature selection.
Drops features where x_obs (TomTom) falls outside the simulator's range.
Uses only features where real-world and simulated data overlap.

Usage: python train_snpe_robust.py
Input:  sbi_thetas.pt, sbi_xs.pt, x_obs.pt, stat_names.txt
Output: snpe_posterior.pkl, snpe_samples.pt, diagnostic plots
"""

import pickle
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sbi.inference import NPE
from sbi.utils import BoxUniform
from sbi.analysis import pairplot
from scipy.stats import gaussian_kde

# ============================================================
# CONFIG
# ============================================================
PARAM_NAMES = ["speedFactor", "speedDev", "sigma", "tau"]
PRIOR_LOW = torch.tensor([0.5, 0.0, 0.0, 0.5], dtype=torch.float32)
PRIOR_HIGH = torch.tensor([1.3, 0.3, 1.0, 3.0], dtype=torch.float32)
N_POSTERIOR_SAMPLES = 50_000
N_CALIB_SIMS = 200

# Feature selection: how far outside training range is acceptable
# Features where x_obs is more than TOLERANCE * std outside range are dropped
TOLERANCE_STD = 0.5  # conservative: only keep features that clearly overlap

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 65)
print("  SNPE TRAINING — ROBUST FEATURE SELECTION")
print("=" * 65)

theta = torch.load("sbi_thetas.pt", weights_only=True).float()
x = torch.load("sbi_xs.pt", weights_only=True).float()
x_obs = torch.load("x_obs.pt", weights_only=True).float()

# Load stat names
stat_names = []
if Path("stat_names.txt").exists():
    with open("stat_names.txt") as f:
        stat_names = [line.strip() for line in f if line.strip()]
if len(stat_names) != x.shape[1]:
    stat_names = [f"feat_{j}" for j in range(x.shape[1])]

print(f"\nRaw data loaded:")
print(f"  theta:  {theta.shape}")
print(f"  x:      {x.shape}")
print(f"  x_obs:  {x_obs.shape}")

# ============================================================
# FEATURE SELECTION
# ============================================================
print("\n" + "=" * 65)
print("  FEATURE SELECTION")
print("=" * 65)

n_feats = x.shape[1]
x_np = x.numpy()
x_obs_np = x_obs.numpy()

keep_indices = []
drop_indices = []
drop_reasons = []

for j in range(n_feats):
    col = x_np[:, j]
    obs_val = x_obs_np[j]
    col_min = col.min()
    col_max = col.max()
    col_std = col.std()
    col_mean = col.mean()

    # Drop constant columns
    if col_std < 1e-8:
        drop_indices.append(j)
        drop_reasons.append(f"constant (std={col_std:.2e})")
        continue

    # Check if x_obs is in range (with tolerance)
    margin = TOLERANCE_STD * col_std

    if obs_val < col_min - margin:
        distance = (col_min - obs_val) / col_std
        drop_indices.append(j)
        drop_reasons.append(f"x_obs BELOW by {distance:.1f}σ")
    elif obs_val > col_max + margin:
        distance = (obs_val - col_max) / col_std
        drop_indices.append(j)
        drop_reasons.append(f"x_obs ABOVE by {distance:.1f}σ")
    else:
        keep_indices.append(j)

print(f"\nFeature selection results:")
print(f"  Total features:  {n_feats}")
print(f"  Kept:            {len(keep_indices)}")
print(f"  Dropped:         {len(drop_indices)}")

print(f"\n  Kept features:")
for j in keep_indices:
    obs_val = x_obs_np[j]
    col = x_np[:, j]
    # Where does x_obs fall as a percentile?
    pct = (col < obs_val).mean() * 100
    print(f"    {j:3d} {stat_names[j]:>25}: x_obs={obs_val:.4f}  "
          f"train=[{col.min():.4f}, {col.max():.4f}]  pctile={pct:.0f}%")

print(f"\n  Dropped features:")
for j, reason in zip(drop_indices, drop_reasons):
    obs_val = x_obs_np[j]
    col = x_np[:, j]
    print(f"    {j:3d} {stat_names[j]:>25}: x_obs={obs_val:.4f}  "
          f"train=[{col.min():.4f}, {col.max():.4f}]  reason={reason}")

# Apply feature selection
keep_mask = torch.tensor(keep_indices)
x_selected = x[:, keep_mask]
x_obs_selected = x_obs[keep_mask]
selected_names = [stat_names[j] for j in keep_indices]

print(f"\nSelected data shapes:")
print(f"  x:     {x_selected.shape}")
print(f"  x_obs: {x_obs_selected.shape}")

# Save feature selection info
torch.save({
    "keep_indices": keep_indices,
    "drop_indices": drop_indices,
    "drop_reasons": drop_reasons,
    "selected_names": selected_names,
    "keep_mask": keep_mask,
}, "snpe_feature_selection.pt")
print("  Saved: snpe_feature_selection.pt")

# ============================================================
# VERIFY x_obs IS NOW IN RANGE
# ============================================================
print("\n--- Verifying x_obs is within training range ---")
x_sel_np = x_selected.numpy()
x_obs_sel_np = x_obs_selected.numpy()
still_out = 0
for j_new in range(len(keep_indices)):
    col = x_sel_np[:, j_new]
    obs = x_obs_sel_np[j_new]
    if obs < col.min() or obs > col.max():
        still_out += 1
        pname = selected_names[j_new]
        print(f"  WARNING: {pname} still slightly out of range: "
              f"x_obs={obs:.4f}, train=[{col.min():.4f}, {col.max():.4f}]")

if still_out == 0:
    print("  All selected features: x_obs within training range ✓")
else:
    print(f"  {still_out} features still marginally out of range (within tolerance)")

# ============================================================
# CHECK FEATURE INFORMATIVENESS
# ============================================================
print("\n--- Feature-parameter correlations ---")
print(f"{'Feature':>25} |", end="")
for name in PARAM_NAMES:
    print(f" {name:>12} |", end="")
print()
print("-" * 80)

theta_np = theta.numpy()
informative_features = []
for j_new in range(len(keep_indices)):
    fname = selected_names[j_new]
    col = x_sel_np[:, j_new]
    print(f"{fname:>25} |", end="")
    max_corr = 0
    for i in range(4):
        corr = np.corrcoef(theta_np[:, i], col)[0, 1]
        max_corr = max(max_corr, abs(corr))
        marker = " *" if abs(corr) > 0.3 else "  "
        print(f" {corr:11.3f}{marker}|", end="")
    print()
    if max_corr > 0.05:
        informative_features.append(j_new)

print(f"\nFeatures with |corr| > 0.3 to any param: marked with *")
print(f"Informative features (|corr| > 0.05): {len(informative_features)}/{len(keep_indices)}")

# ============================================================
# ADDITIONAL: Remove highly correlated features (optional)
# ============================================================
print("\n--- Checking inter-feature correlation ---")
corr_matrix = np.corrcoef(x_sel_np.T)
np.fill_diagonal(corr_matrix, 0)
high_corr_pairs = []
for i in range(len(keep_indices)):
    for j in range(i + 1, len(keep_indices)):
        if abs(corr_matrix[i, j]) > 0.98:
            high_corr_pairs.append((i, j, corr_matrix[i, j]))

print(f"Highly correlated pairs (|r| > 0.98): {len(high_corr_pairs)}")
if high_corr_pairs:
    for i, j, r in high_corr_pairs[:10]:
        print(f"  {selected_names[i]} <-> {selected_names[j]}: r={r:.4f}")

# ============================================================
# PRIOR
# ============================================================
prior = BoxUniform(low=PRIOR_LOW, high=PRIOR_HIGH)

# ============================================================
# TRAIN NPE
# ============================================================
print("\n" + "=" * 65)
print("  TRAINING NPE")
print("=" * 65)

inference = NPE(prior=prior, density_estimator="nsf")
inference.append_simulations(theta, x_selected)

print(f"Training NPE (NSF) with:")
print(f"  Simulations:   {len(theta)}")
print(f"  Features:      {x_selected.shape[1]} (from original {n_feats})")
print(f"  Parameters:    {len(PARAM_NAMES)}")
print()

t_train_start = time.time()

density_estimator = inference.train(
    training_batch_size=128,
    learning_rate=5e-4,
    max_num_epochs=500,
    stop_after_epochs=30,
    clip_max_norm=5.0,
    show_train_summary=True,
)

train_time = time.time() - t_train_start
print(f"\nTraining completed in {train_time:.1f}s ({train_time/60:.1f} min)")

posterior = inference.build_posterior(density_estimator)
print("Posterior built successfully")

# ============================================================
# INFERENCE ON x_obs
# ============================================================
print("\n" + "=" * 65)
print("  INFERENCE ON OBSERVED DATA (TomTom)")
print("=" * 65)

print(f"Sampling {N_POSTERIOR_SAMPLES} posterior samples...")
t_sample_start = time.time()
samples = posterior.sample((N_POSTERIOR_SAMPLES,), x=x_obs_selected.unsqueeze(0))
sample_time = time.time() - t_sample_start
print(f"Sampling done in {sample_time:.1f}s")

samples_np = samples.numpy()

# Point estimates
print(f"\n{'Parameter':>15} | {'Mean':>8} | {'Median':>8} | {'Std':>8} | "
      f"{'5%':>8} | {'95%':>8} | {'MAP':>8}")
print("-" * 80)

map_estimates = []
for i, name in enumerate(PARAM_NAMES):
    col = samples_np[:, i]
    mean = col.mean()
    median = np.median(col)
    std = col.std()
    q05, q95 = np.percentile(col, [5, 95])

    try:
        kde = gaussian_kde(col)
        grid = np.linspace(PRIOR_LOW[i].item(), PRIOR_HIGH[i].item(), 1000)
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

# 1. Pairplot
print("[1/5] Pairplot...")
fig, axes = pairplot(
    samples_np,
    labels=PARAM_NAMES,
    limits=list(zip(PRIOR_LOW.numpy(), PRIOR_HIGH.numpy())),
    figsize=(10, 10),
)
fig.suptitle("SNPE Posterior (robust features, TomTom x_obs)", fontsize=14, y=1.02)
plt.savefig("snpe_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: snpe_pairplot.png")

# 2. Marginal posteriors
print("[2/5] Marginals...")
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for i, (name, ax) in enumerate(zip(PARAM_NAMES, axes)):
    col = samples_np[:, i]
    lo, hi = PRIOR_LOW[i].item(), PRIOR_HIGH[i].item()

    ax.hist(col, bins=80, density=True, alpha=0.7, color="steelblue",
            edgecolor="white", linewidth=0.3, label="Posterior")
    ax.axhline(1.0 / (hi - lo), color="gray", linestyle="--",
               alpha=0.6, label="Prior")
    ax.axvline(col.mean(), color="red", linewidth=1.5,
               label=f"Mean={col.mean():.3f}")
    q05, q95 = np.percentile(col, [5, 95])
    ax.axvspan(q05, q95, alpha=0.15, color="red", label="90% CI")

    ax.set_xlabel(name, fontsize=12)
    ax.set_ylabel("Density" if i == 0 else "")
    ax.set_xlim(lo, hi)
    ax.legend(fontsize=8)
    ax.set_title(name)

plt.suptitle("SNPE Marginal Posteriors — Robust Features (TomTom)", fontsize=13)
plt.tight_layout()
plt.savefig("snpe_marginals.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: snpe_marginals.png")

# 3. SBC calibration check
print(f"[3/5] Calibration check ({N_CALIB_SIMS} sims)...")
np.random.seed(42)
calib_indices = np.random.choice(len(theta), N_CALIB_SIMS, replace=False)

ci_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
coverage_by_level = {level: [] for level in ci_levels}
recovery_errors = []

for k, idx in enumerate(calib_indices):
    x_test = x_selected[idx].unsqueeze(0)
    true_theta = theta[idx].numpy()

    try:
        samps = posterior.sample((2000,), x=x_test).numpy()
    except Exception:
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
ax.fill_between([0.45, 1], [0.4, 0.95], [0.5, 1.05],
                alpha=0.1, color="gray", label="±5% band")
ax.set_xlabel("Expected coverage", fontsize=12)
ax.set_ylabel("Observed coverage", fontsize=12)
ax.set_title("Simulation-Based Calibration (internal)", fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(0.45, 1.0)
ax.set_ylim(0.45, 1.0)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("snpe_calibration.png", dpi=150)
plt.close()
print("  Saved: snpe_calibration.png")

# 4. Recovery bias
print("[4/5] Recovery bias...")
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for i, (name, ax) in enumerate(zip(PARAM_NAMES, axes)):
    errors = [r["error"] for r in recovery_errors if r["param"] == i]
    ax.hist(errors, bins=40, alpha=0.7, color="coral", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
    mean_err = np.mean(errors)
    ax.axvline(mean_err, color="red", linewidth=1.5,
               label=f"Bias={mean_err:.4f}")
    ax.set_xlabel(f"Post. mean - True ({name})")
    ax.set_ylabel("Count" if i == 0 else "")
    ax.legend(fontsize=8)
    ax.set_title(name)

plt.suptitle("Posterior Recovery Bias (centered at 0 = good)", fontsize=13)
plt.tight_layout()
plt.savefig("snpe_recovery_bias.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: snpe_recovery_bias.png")

# 5. Feature importance: which features drove the posterior?
print("[5/5] Feature importance (sensitivity)...")
fig, ax = plt.subplots(figsize=(14, 6))

# Compute: how much does posterior mean shift when we perturb each feature?
baseline_mean = samples_np.mean(axis=0)
importance = np.zeros((len(keep_indices), 4))

for j_new in range(len(keep_indices)):
    x_perturbed = x_obs_selected.clone()
    # Perturb to training mean
    x_perturbed[j_new] = x_selected[:, j_new].mean()
    try:
        perturbed_samples = posterior.sample((5000,), x=x_perturbed.unsqueeze(0)).numpy()
        for i in range(4):
            importance[j_new, i] = abs(perturbed_samples[:, i].mean() - baseline_mean[i])
    except Exception:
        pass

# Total importance per feature
total_importance = importance.sum(axis=1)
sorted_idx = np.argsort(total_importance)[::-1]

# Bar chart
top_n = min(20, len(keep_indices))
top_idx = sorted_idx[:top_n]
y_pos = np.arange(top_n)

colors_param = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
bottoms = np.zeros(top_n)
for i, (pname, color) in enumerate(zip(PARAM_NAMES, colors_param)):
    vals = importance[top_idx, i]
    ax.barh(y_pos, vals, left=bottoms, color=color, alpha=0.8, label=pname)
    bottoms += vals

ax.set_yticks(y_pos)
ax.set_yticklabels([selected_names[j] for j in top_idx], fontsize=9)
ax.set_xlabel("Mean absolute shift in posterior mean")
ax.set_title("Feature Importance (which features drive the posterior)")
ax.legend(fontsize=9)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("snpe_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: snpe_feature_importance.png")

# ============================================================
# SAVE EVERYTHING
# ============================================================
print("\n" + "=" * 65)
print("  SAVING")
print("=" * 65)

with open("snpe_posterior.pkl", "wb") as f:
    pickle.dump(posterior, f)
print("  snpe_posterior.pkl")

torch.save(samples, "snpe_samples.pt")
print("  snpe_samples.pt")

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
    "n_features_original": n_feats,
    "n_features_selected": len(keep_indices),
    "kept_feature_names": selected_names,
    "total_sims": len(theta),
}
np.savez("snpe_results.npz", **{k: v for k, v in results.items()
                                  if not isinstance(v, list)},
         kept_feature_names=selected_names,
         calibration_expected=expected_cov,
         calibration_observed=observed_cov)
print("  snpe_results.npz")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 65)
print("  RESULTS SUMMARY")
print("=" * 65)
print(f"\nTraining:   {len(theta)} sims, {train_time:.0f}s")
print(f"Features:   {len(keep_indices)} / {n_feats} (dropped {len(drop_indices)} incompatible)")
print(f"Sampling:   {N_POSTERIOR_SAMPLES} posterior samples, {sample_time:.1f}s")
print(f"\nInferred parameters (from TomTom x_obs):")
print(f"{'Parameter':>15} | {'MAP':>8} | {'Mean±Std':>15} | {'90% CI':>18}")
print("-" * 65)
for i, name in enumerate(PARAM_NAMES):
    col = samples_np[:, i]
    q05, q95 = np.percentile(col, [5, 95])
    print(f"{name:>15} | {map_estimates[i]:8.4f} | "
          f"{col.mean():7.4f} ± {col.std():.4f} | "
          f"[{q05:.4f}, {q95:.4f}]")

print(f"\nCalibration (internal SBC):")
max_dev = max(abs(e - o) for e, o in zip(expected_cov, observed_cov))
if max_dev < 0.05:
    cal_status = "EXCELLENT"
elif max_dev < 0.10:
    cal_status = "GOOD"
elif max_dev < 0.15:
    cal_status = "ACCEPTABLE"
else:
    cal_status = "POOR"
print(f"  {cal_status} (max deviation: {max_dev:.1%})")

print(f"\nNOTE: Calibration is 'internal' — checks if posterior can recover")
print(f"sim parameters from sim data. Since x_obs is real-world (TomTom),")
print(f"there may be an additional sim-to-real gap not captured here.")

print(f"\nFiles:")
print(f"  snpe_posterior.pkl          — trained posterior")
print(f"  snpe_samples.pt             — {N_POSTERIOR_SAMPLES} posterior samples")
print(f"  snpe_results.npz            — point estimates + calibration")
print(f"  snpe_feature_selection.pt   — which features were kept/dropped")
print(f"  snpe_pairplot.png           — joint posterior")
print(f"  snpe_marginals.png          — marginal posteriors vs prior")
print(f"  snpe_calibration.png        — internal SBC calibration")
print(f"  snpe_recovery_bias.png      — parameter recovery bias")
print(f"  snpe_feature_importance.png — which features drive inference")
print(f"\nNext: python train_asnpe.py  (uses same robust features)")


if __name__ == "__main__":
    pass  # All code runs at module level for simplicity