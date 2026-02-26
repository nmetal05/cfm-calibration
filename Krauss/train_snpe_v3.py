"""
Train SNPE — v3: Deduplicated features + MCMC fallback

Key fixes:
1. Strict feature filtering (must be IN training range)
2. Remove redundant features (|r| > 0.95)
3. MCMC sampling fallback if rejection fails

Usage: python train_snpe_v3.py
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

# Correlation threshold for deduplication
CORR_THRESHOLD = 0.95

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 65)
print("  SNPE v3 — DEDUPLICATED + STRICT FILTERING")
print("=" * 65)

theta_full = torch.load("sbi_thetas.pt", weights_only=True).float()
x = torch.load("sbi_xs.pt", weights_only=True).float()
x_obs = torch.load("x_obs.pt", weights_only=True).float()

theta = theta_full[:1000]
x = x[:1000]

print(f"\nBudget adjusted: Using 1000 simulations (down from {len(theta_full)})")
# ======
stat_names = []
if Path("stat_names.txt").exists():
    with open("stat_names.txt") as f:
        stat_names = [line.strip() for line in f if line.strip()]
if len(stat_names) != x.shape[1]:
    stat_names = [f"feat_{j}" for j in range(x.shape[1])]

print(f"\nRaw: theta={theta.shape}, x={x.shape}, x_obs={x_obs.shape}")

# ============================================================
# STEP 1: STRICT feature filtering
# Only keep features where x_obs is INSIDE training [min, max]
# ============================================================
print("\n" + "=" * 65)
print("  STEP 1: STRICT RANGE FILTERING")
print("=" * 65)

x_np = x.numpy()
x_obs_np = x_obs.numpy()

range_ok = []
range_dropped = []

for j in range(x.shape[1]):
    col = x_np[:, j]
    obs = x_obs_np[j]
    col_std = col.std()

    # Drop constant
    if col_std < 1e-8:
        range_dropped.append((j, stat_names[j], "constant"))
        continue

    # STRICT: must be within [min, max] of training data
    if obs < col.min() or obs > col.max():
        dist = max(col.min() - obs, obs - col.max()) / col_std
        range_dropped.append((j, stat_names[j], f"out of range ({dist:.1f}σ)"))
        continue

    range_ok.append(j)

print(f"  After range filter: {len(range_ok)} kept, {len(range_dropped)} dropped")
for j, name, reason in range_dropped:
    print(f"    DROP {j:3d} {name:>25}: {reason}")

# ============================================================
# STEP 2: Remove redundant features (high correlation)
# ============================================================
print("\n" + "=" * 65)
print("  STEP 2: DEDUPLICATION (|r| > {:.2f})".format(CORR_THRESHOLD))
print("=" * 65)

x_filtered = x_np[:, range_ok]
filtered_names = [stat_names[j] for j in range_ok]
theta_np = theta.numpy()

# Compute correlation matrix among remaining features
corr = np.corrcoef(x_filtered.T)

# Greedy selection: keep features that aren't too correlated with already-kept ones
# Prioritize features with highest correlation to ANY parameter
param_importance = np.zeros(len(range_ok))
for j_new in range(len(range_ok)):
    for i in range(4):
        r = abs(np.corrcoef(theta_np[:, i], x_filtered[:, j_new])[0, 1])
        param_importance[j_new] = max(param_importance[j_new], r)

# Sort by importance (most informative first)
order = np.argsort(-param_importance)

kept_new_indices = []
kept_original_indices = []

for idx in order:
    # Check if this feature is too correlated with any already-kept feature
    too_correlated = False
    for kept_idx in kept_new_indices:
        if abs(corr[idx, kept_idx]) > CORR_THRESHOLD:
            too_correlated = True
            break

    if not too_correlated:
        kept_new_indices.append(idx)
        kept_original_indices.append(range_ok[idx])

kept_new_indices.sort()
kept_original_indices.sort()

final_names = [stat_names[j] for j in kept_original_indices]
n_removed = len(range_ok) - len(kept_new_indices)

print(f"  After deduplication: {len(kept_original_indices)} kept, {n_removed} removed")
print(f"\n  Final features ({len(kept_original_indices)}):")

# Show which features were kept and their parameter correlations
print(f"  {'Idx':>4} {'Name':>25} | {'sFact':>7} {'sDev':>7} {'sigma':>7} {'tau':>7} | {'x_obs':>8} {'pctile':>7}")
print(f"  " + "-" * 95)

x_final = x_np[:, kept_original_indices]
x_obs_final_np = x_obs_np[kept_original_indices]

for k, j_orig in enumerate(kept_original_indices):
    name = stat_names[j_orig]
    obs = x_obs_np[j_orig]
    col = x_np[:, j_orig]
    pct = (col < obs).mean() * 100

    corrs = []
    for i in range(4):
        r = np.corrcoef(theta_np[:, i], col)[0, 1]
        corrs.append(r)

    print(f"  {j_orig:4d} {name:>25} | {corrs[0]:+7.3f} {corrs[1]:+7.3f} "
          f"{corrs[2]:+7.3f} {corrs[3]:+7.3f} | {obs:8.4f} {pct:6.0f}%")

# Show removed (deduplicated) features
dedup_removed = [range_ok[idx] for idx in range(len(range_ok))
                 if idx not in kept_new_indices]
print(f"\n  Removed by deduplication ({n_removed}):")
for j_orig in dedup_removed:
    name = stat_names[j_orig]
    # Find which kept feature it's most correlated with
    j_in_filtered = range_ok.index(j_orig)
    best_corr = 0
    best_match = ""
    for k_idx in kept_new_indices:
        r = abs(corr[j_in_filtered, k_idx])
        if r > best_corr:
            best_corr = r
            best_match = filtered_names[k_idx]
    print(f"    {j_orig:3d} {name:>25}  (r={best_corr:.3f} with {best_match})")

# ============================================================
# STEP 3: Verify and prepare tensors
# ============================================================
print("\n" + "=" * 65)
print("  STEP 3: FINAL VERIFICATION")
print("=" * 65)

x_final_t = torch.tensor(x_final, dtype=torch.float32)
x_obs_final = torch.tensor(x_obs_final_np, dtype=torch.float32)

# Verify ALL features in range
all_in_range = True
for k in range(x_final.shape[1]):
    col = x_final[:, k]
    obs = x_obs_final_np[k]
    if obs < col.min() or obs > col.max():
        print(f"  WARNING: {final_names[k]} still out of range!")
        all_in_range = False

if all_in_range:
    print(f"  ✓ All {x_final.shape[1]} features: x_obs within training range")

# Check correlation matrix of final features
final_corr = np.corrcoef(x_final.T)
np.fill_diagonal(final_corr, 0)
max_corr = np.abs(final_corr).max()
print(f"  ✓ Max inter-feature correlation: {max_corr:.3f} (threshold was {CORR_THRESHOLD})")

# Where does x_obs sit in the training distribution?
percentiles = [(x_final[:, k] < x_obs_final_np[k]).mean() for k in range(x_final.shape[1])]
print(f"  x_obs percentile range: [{min(percentiles)*100:.0f}%, {max(percentiles)*100:.0f}%]")
print(f"  x_obs mean percentile:  {np.mean(percentiles)*100:.0f}%")

print(f"\n  Final shapes:")
print(f"    theta:  {theta.shape}")
print(f"    x:      {x_final_t.shape}")
print(f"    x_obs:  {x_obs_final.shape}")

# Save feature selection
torch.save({
    "kept_original_indices": kept_original_indices,
    "final_names": final_names,
    "n_original": x.shape[1],
    "range_dropped": range_dropped,
    "dedup_removed": dedup_removed,
}, "snpe_feature_selection.pt")

# ============================================================
# TRAIN NPE
# ============================================================
print("\n" + "=" * 65)
print("  TRAINING NPE")
print("=" * 65)

prior = BoxUniform(low=PRIOR_LOW, high=PRIOR_HIGH)

inference = NPE(prior=prior, density_estimator="nsf")
inference.append_simulations(theta, x_final_t)

print(f"  Sims: {len(theta)}, Features: {x_final_t.shape[1]}, Params: 4")
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
print(f"\nTraining: {train_time:.1f}s ({train_time/60:.1f} min)")

# ============================================================
# BUILD POSTERIOR — try direct first, fall back to MCMC
# ============================================================
print("\n" + "=" * 65)
print("  INFERENCE")
print("=" * 65)

# Try direct sampling first (fast)
print("Attempting direct (rejection) sampling...")
posterior = inference.build_posterior(density_estimator)

try:
    # Quick test: try to get 100 samples in 30 seconds
    import signal
    test_samples = []
    t_test = time.time()
    test_batch = posterior.sample((100,), x=x_obs_final.unsqueeze(0))
    test_time = time.time() - t_test

    if test_time < 30:
        print(f"  Direct sampling works! ({test_time:.1f}s for 100 samples)")
        sampling_method = "direct"
    else:
        raise TimeoutError("Too slow")

except Exception as e:
    print(f"  Direct sampling too slow or failed: {e}")
    print("  Switching to MCMC sampling...")

    posterior = inference.build_posterior(
        density_estimator,
        sample_with="mcmc",
        mcmc_method="slice_np_vectorized",
        mcmc_parameters={
            "num_chains": 20,
            "thin": 10,
            "warmup_steps": 200,
            "init_strategy": "proposal",
        },
    )
    sampling_method = "mcmc"
    print("  MCMC posterior built")

# Sample
print(f"\nSampling {N_POSTERIOR_SAMPLES} samples via {sampling_method}...")
t_sample_start = time.time()
samples = posterior.sample((N_POSTERIOR_SAMPLES,), x=x_obs_final.unsqueeze(0))
sample_time = time.time() - t_sample_start
print(f"  Done in {sample_time:.1f}s")

samples_np = samples.numpy()

# Clip to prior bounds
for i in range(4):
    samples_np[:, i] = np.clip(samples_np[:, i], PRIOR_LOW[i].item(), PRIOR_HIGH[i].item())

# ============================================================
# RESULTS
# ============================================================
print(f"\n{'Parameter':>15} | {'Mean':>8} | {'Median':>8} | {'Std':>8} | "
      f"{'5%':>8} | {'95%':>8} | {'MAP':>8}")
print("-" * 80)

map_estimates = []
for i, name in enumerate(PARAM_NAMES):
    col = samples_np[:, i]
    q05, q95 = np.percentile(col, [5, 95])
    try:
        kde = gaussian_kde(col)
        grid = np.linspace(PRIOR_LOW[i].item(), PRIOR_HIGH[i].item(), 1000)
        map_val = grid[kde(grid).argmax()]
    except Exception:
        map_val = np.median(col)
    map_estimates.append(map_val)

    print(f"{name:>15} | {col.mean():8.4f} | {np.median(col):8.4f} | {col.std():8.4f} | "
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
fig.suptitle(f"SNPE Posterior ({sampling_method}, {x_final_t.shape[1]} features)", fontsize=14, y=1.02)
plt.savefig("snpe_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: snpe_pairplot.png")

# 2. Marginals
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

plt.suptitle("SNPE Marginal Posteriors — TomTom x_obs", fontsize=13)
plt.tight_layout()
plt.savefig("snpe_marginals.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: snpe_marginals.png")

# 3. SBC Calibration
print(f"[3/5] Calibration ({N_CALIB_SIMS} sims)...")
np.random.seed(42)
calib_indices = np.random.choice(len(theta), N_CALIB_SIMS, replace=False)

ci_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
coverage_by_level = {level: [] for level in ci_levels}
recovery_errors = []

for k, idx in enumerate(calib_indices):
    x_test = x_final_t[idx].unsqueeze(0)
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
        print(f"    {k+1}/{N_CALIB_SIMS}")

print(f"\n  Calibration:")
expected_cov = []
observed_cov = []
for level in ci_levels:
    obs = np.mean(coverage_by_level[level])
    expected_cov.append(level)
    observed_cov.append(obs)
    status = "OK" if abs(obs - level) < 0.1 else "WARN"
    print(f"    {level*100:4.0f}% CI: observed {obs*100:5.1f}%  [{status}]")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Ideal")
ax.plot(expected_cov, observed_cov, "o-", color="steelblue", markersize=8, linewidth=2, label="SNPE")
ax.set_xlabel("Expected coverage")
ax.set_ylabel("Observed coverage")
ax.set_title("Simulation-Based Calibration")
ax.legend()
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
    ax.axvline(mean_err, color="red", linewidth=1.5, label=f"Bias={mean_err:.4f}")
    ax.set_xlabel(f"Post. mean - True ({name})")
    ax.legend(fontsize=8)
    ax.set_title(name)

plt.suptitle("Posterior Recovery Bias", fontsize=13)
plt.tight_layout()
plt.savefig("snpe_recovery_bias.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: snpe_recovery_bias.png")

# 5. Feature importance
print("[5/5] Feature importance...")
baseline_mean = samples_np.mean(axis=0)
importance = np.zeros((len(kept_original_indices), 4))

for j_new in range(len(kept_original_indices)):
    x_perturbed = x_obs_final.clone()
    x_perturbed[j_new] = x_final_t[:, j_new].mean()
    try:
        psamps = posterior.sample((5000,), x=x_perturbed.unsqueeze(0)).numpy()
        for i in range(4):
            importance[j_new, i] = abs(psamps[:, i].mean() - baseline_mean[i])
    except Exception:
        pass

total_importance = importance.sum(axis=1)
sorted_idx = np.argsort(total_importance)[::-1]

fig, ax = plt.subplots(figsize=(12, 6))
top_n = min(15, len(kept_original_indices))
top_idx = sorted_idx[:top_n]
y_pos = np.arange(top_n)

colors_param = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
bottoms = np.zeros(top_n)
for i, (pname, color) in enumerate(zip(PARAM_NAMES, colors_param)):
    vals = importance[top_idx, i]
    ax.barh(y_pos, vals, left=bottoms, color=color, alpha=0.8, label=pname)
    bottoms += vals

ax.set_yticks(y_pos)
ax.set_yticklabels([final_names[j] for j in top_idx], fontsize=9)
ax.set_xlabel("Mean absolute shift in posterior mean")
ax.set_title("Feature Importance")
ax.legend(fontsize=9)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("snpe_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: snpe_feature_importance.png")

# ============================================================
# SAVE
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
    "sample_time_s": sample_time,
    "sampling_method": sampling_method,
    "n_sims": len(theta),
    "n_features": len(kept_original_indices),
    "total_sims": len(theta),
}
np.savez("snpe_results.npz", **results)
print("  snpe_results.npz")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)
print(f"\n  Features:  {len(kept_original_indices)} (from {x.shape[1]} original)")
print(f"  Training:  {train_time:.0f}s")
print(f"  Sampling:  {sampling_method}, {N_POSTERIOR_SAMPLES} samples in {sample_time:.1f}s")

print(f"\n  Inferred parameters (TomTom x_obs):")
print(f"  {'Parameter':>15} | {'MAP':>8} | {'Mean±Std':>15} | {'90% CI':>18}")
print(f"  " + "-" * 60)
for i, name in enumerate(PARAM_NAMES):
    col = samples_np[:, i]
    q05, q95 = np.percentile(col, [5, 95])
    print(f"  {name:>15} | {map_estimates[i]:8.4f} | "
          f"{col.mean():7.4f} ± {col.std():.4f} | [{q05:.4f}, {q95:.4f}]")

max_dev = max(abs(e - o) for e, o in zip(expected_cov, observed_cov))
cal = "EXCELLENT" if max_dev < 0.05 else "GOOD" if max_dev < 0.10 else "ACCEPTABLE" if max_dev < 0.15 else "POOR"
print(f"\n  Calibration: {cal} (max deviation {max_dev:.1%})")

print(f"\n  Interpretation:")
print(f"  - speedFactor: how fast drivers go relative to speed limit")
print(f"  - speedDev:    driver-to-driver speed variation")
print(f"  - sigma:       driver imperfection / randomness (Krauss)")
print(f"  - tau:         desired time headway (seconds)")
print(f"\n  Next: python train_asnpe.py")