"""
Train ASNPE — Active Sequential Neural Posterior Estimation
Uses same deduplicated features as SNPE v3.

Usage: python train_asnpe.py
Input:  osm.net.xml, sbi_peak.sumocfg, x_obs.pt, snpe_feature_selection.pt
Output: asnpe_posterior.pkl, asnpe_samples.pt, diagnostic plots
"""

import os
import time
import uuid
import pickle
import subprocess
import traceback
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from seqinf.methods.posterior import ASNPE
from seqinf import BayesianInferenceDiagnostic
from sbi.utils import BoxUniform
from sbi.analysis import pairplot
from scipy.stats import gaussian_kde

from write_vtype import write_vtype_file
from parse_output_shared import (
    parse_edge_data,
    summary_statistics_sumo,
    get_edge_max_speeds,
)

# ============================================================
# CONFIG
# ============================================================
SUMO_BINARY = "sumo"
BASE_DIR = Path(".")
SIM_TIMEOUT = 120

# ASNPE budget — same total as SNPE (5000) for fair comparison
# Also test with fewer sims to show sample efficiency
N_ROUNDS = 10
N_SAMPLES_PER_ROUND = 50  # 10 × 500 = 5000 total (fair comparison)
N_WORKERS = 15
N_POSTERIOR_SAMPLES = 50_000

PARAM_NAMES = ["speedFactor", "speedDev", "sigma", "tau"]
PRIOR_LOW = torch.tensor([0.5, 0.0, 0.0, 0.5], dtype=torch.float32)
PRIOR_HIGH = torch.tensor([1.3, 0.3, 1.0, 3.0], dtype=torch.float32)

# ============================================================
# LOAD NETWORK + FEATURE SELECTION
# ============================================================
print("=" * 65)
print("  ASNPE — Active Sequential Neural Posterior Estimation")
print("=" * 65)

print("\nLoading network edge speeds...")
EDGE_MAX_SPEEDS, EDGE_LENGTHS = get_edge_max_speeds("osm.net.xml")
print(f"  Loaded {len(EDGE_MAX_SPEEDS)} edges")

# Load feature selection from SNPE v3
print("\nLoading feature selection from SNPE v3...")
feat_sel = torch.load("snpe_feature_selection.pt", weights_only=False)
KEPT_INDICES = feat_sel["kept_original_indices"]
FEAT_NAMES = feat_sel["final_names"]
print(f"  Using {len(KEPT_INDICES)} features (from 57 original)")
print(f"  Features: {FEAT_NAMES}")

# Load and select x_obs
x_obs_full = torch.load("x_obs.pt", weights_only=True).float()
x_obs = x_obs_full[KEPT_INDICES]
print(f"  x_obs shape: {x_obs.shape}")

# Global sim counter
_sim_counter = {"count": 0, "fail": 0, "start_time": None}


# ============================================================
# SUMO SIMULATOR — returns SELECTED features only
# ============================================================
def sumo_simulator(theta):
    """
    SUMO simulator for seqinf ASNPE.
    Returns only the 11 selected summary statistics.

    Parameters
    ----------
    theta : Tensor, shape (4,)

    Returns
    -------
    x : Tensor, shape (11,)
    """
    if _sim_counter["start_time"] is None:
        _sim_counter["start_time"] = time.time()

    theta_np = theta.numpy() if isinstance(theta, torch.Tensor) else np.array(theta)

    sim_id = uuid.uuid4().hex[:12]
    sim_dir = BASE_DIR / "asnpe_runs" / f"sim_{sim_id}"
    sim_dir.mkdir(parents=True, exist_ok=True)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            vtype_path = sim_dir / "vtype.xml"
            write_vtype_file(theta_np.tolist(), str(vtype_path))

            edgedata_output = sim_dir / "edgedata.xml"
            edgedata_add = sim_dir / "edgedata.add.xml"
            with open(edgedata_add, "w", encoding="utf-8") as f:
                f.write(
                    "<additional>\n"
                    f'    <edgeData id="asnpe" freq="300" '
                    f'file="{edgedata_output.resolve()}" excludeEmpty="true"/>\n'
                    "</additional>\n"
                )

            if edgedata_output.exists():
                edgedata_output.unlink()

            result = subprocess.run(
                [
                    SUMO_BINARY,
                    "-c", str((BASE_DIR / "sbi_peak.sumocfg").resolve()),
                    "--additional-files",
                    f"{vtype_path.resolve()},{edgedata_add.resolve()}",
                    "--seed", str(np.random.randint(100000)),
                ],
                capture_output=True,
                timeout=SIM_TIMEOUT,
                cwd=str(BASE_DIR.resolve()),
            )

            if result.returncode == 0 and edgedata_output.exists():
                edge_data = parse_edge_data(str(edgedata_output))
                x_full = summary_statistics_sumo(edge_data, EDGE_MAX_SPEEDS, EDGE_LENGTHS)

                _cleanup_sim_dir(sim_dir)

                if x_full is not None and len(x_full) == 57 and np.all(np.isfinite(x_full)):
                    # Apply feature selection — return only 11 features
                    x_selected = x_full[KEPT_INDICES]

                    _sim_counter["count"] += 1
                    if _sim_counter["count"] % 50 == 0:
                        elapsed = time.time() - _sim_counter["start_time"]
                        rate = _sim_counter["count"] / (elapsed / 60)
                        print(
                            f"    [ASNPE] Sims: {_sim_counter['count']} | "
                            f"Failed: {_sim_counter['fail']} | "
                            f"Rate: {rate:.0f}/min | "
                            f"Elapsed: {elapsed/60:.1f} min"
                        )

                    return torch.tensor(x_selected, dtype=torch.float32)

            if attempt < max_retries - 1:
                time.sleep(0.5)

        except subprocess.TimeoutExpired:
            if attempt < max_retries - 1:
                time.sleep(0.5)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  WARNING: Sim {sim_id} failed: {e}")

    _cleanup_sim_dir(sim_dir)
    _sim_counter["fail"] += 1
    if _sim_counter["fail"] % 10 == 0:
        print(f"  WARNING: {_sim_counter['fail']} total failures")
    return torch.zeros(len(KEPT_INDICES), dtype=torch.float32)


def _cleanup_sim_dir(sim_dir):
    try:
        for f in sim_dir.glob("*"):
            f.unlink(missing_ok=True)
        sim_dir.rmdir()
    except OSError:
        pass


# ============================================================
# MAIN
# ============================================================
def main():
    # Verify setup
    assert (BASE_DIR / "osm.net.xml").exists()
    assert (BASE_DIR / "sbi_peak.sumocfg").exists()
    assert (BASE_DIR / "routes_peak_novtype.rou.xml").exists()
    assert x_obs.shape == (len(KEPT_INDICES),)

    (BASE_DIR / "asnpe_runs").mkdir(exist_ok=True)

    prior = BoxUniform(low=PRIOR_LOW, high=PRIOR_HIGH)

    # --------------------------------------------------------
    # Test simulator
    # --------------------------------------------------------
    print("\nTesting SUMO simulator (with feature selection)...")
    test_theta = torch.tensor([0.9, 0.1, 0.5, 1.5], dtype=torch.float32)
    t0 = time.time()
    test_x = sumo_simulator(test_theta)
    test_time = time.time() - t0
    print(f"  Test sim: {test_time:.1f}s")
    print(f"  Output shape: {test_x.shape} (expect {len(KEPT_INDICES)})")
    print(f"  Output range: [{test_x.min():.4f}, {test_x.max():.4f}]")
    assert test_x.shape == (len(KEPT_INDICES),), f"Wrong shape: {test_x.shape}"
    assert torch.all(torch.isfinite(test_x)), "Non-finite output"
    print("  Simulator OK!")

    # Reset counters
    _sim_counter["count"] = 0
    _sim_counter["fail"] = 0
    _sim_counter["start_time"] = None

    # --------------------------------------------------------
    # ASNPE config
    # --------------------------------------------------------
    total_budget = N_ROUNDS * N_SAMPLES_PER_ROUND
    est_time = total_budget / 38  # based on observed rate

    print(f"\n--- ASNPE Configuration ---")
    print(f"  Rounds:          {N_ROUNDS}")
    print(f"  Samples/round:   {N_SAMPLES_PER_ROUND}")
    print(f"  Total budget:    {total_budget} sims")
    print(f"  Workers:         {N_WORKERS}")
    print(f"  Features:        {len(KEPT_INDICES)} (deduplicated)")
    print(f"  Density est:     MAF with MC-dropout (p=0.5)")
    print(f"  Est. time:       ~{est_time:.0f} min ({est_time/60:.1f} hours)")

    # --------------------------------------------------------
    # Create ASNPE
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("  CREATING ASNPE INFERENCE OBJECT")
    print("=" * 65)

    asnpe = ASNPE(
        simulator=sumo_simulator,
        prior=prior,
        density_estimator="maf",
        num_workers=N_WORKERS,
    )

    print("  ASNPE created successfully")
    print(f"  Collector type: {type(asnpe.collector_cls)}")

    # --------------------------------------------------------
    # Run sequential inference
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("  RUNNING ASNPE")
    print("=" * 65)
    print(f"  Started at {time.strftime('%H:%M:%S')}")
    print()

    t_total_start = time.time()

    asnpe.run(
        num_rounds=N_ROUNDS,
        num_samples=N_SAMPLES_PER_ROUND,
        x_o=x_obs,
        seed=42,
    )

    total_time = time.time() - t_total_start

    print(f"\n  ASNPE completed!")
    print(f"  Total time:      {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    print(f"  Sims completed:  {_sim_counter['count']}")
    print(f"  Sims failed:     {_sim_counter['fail']}")

    # --------------------------------------------------------
    # Extract posterior and sample
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("  EXTRACTING POSTERIOR")
    print("=" * 65)

    posterior = asnpe.proposal
    print(f"  Posterior type: {type(posterior)}")

    print(f"  Sampling {N_POSTERIOR_SAMPLES} posterior samples...")
    t_sample_start = time.time()

    # Try multiple sampling approaches
    sampled = False
    for sample_attempt, sample_kwargs in enumerate([
        {},                                          # no args (already conditioned)
        {"x": x_obs.unsqueeze(0)},                  # with x
        {"x": x_obs},                               # without batch dim
    ]):
        try:
            samples = posterior.sample((N_POSTERIOR_SAMPLES,), **sample_kwargs)
            sampled = True
            print(f"  Sampling succeeded (attempt {sample_attempt + 1})")
            break
        except Exception as e:
            print(f"  Attempt {sample_attempt + 1} failed: {e}")

    if not sampled:
        print("  ERROR: All sampling attempts failed!")
        print("  Saving what we have and exiting...")
        try:
            with open("asnpe_inference.pkl", "wb") as f:
                pickle.dump(asnpe, f)
            print("  Saved: asnpe_inference.pkl (for manual investigation)")
        except Exception:
            pass
        return

    sample_time = time.time() - t_sample_start
    print(f"  Done in {sample_time:.1f}s")

    samples_np = samples.numpy()

    # Clip to prior bounds
    for i in range(4):
        samples_np[:, i] = np.clip(
            samples_np[:, i], PRIOR_LOW[i].item(), PRIOR_HIGH[i].item()
        )

    # --------------------------------------------------------
    # Point estimates
    # --------------------------------------------------------
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

        print(f"{name:>15} | {col.mean():8.4f} | {np.median(col):8.4f} | "
              f"{col.std():8.4f} | {q05:8.4f} | {q95:8.4f} | {map_val:8.4f}")

    # --------------------------------------------------------
    # Per-round statistics
    # --------------------------------------------------------
    print("\n--- Per-Round Statistics ---")
    print(f"{'Round':>5} | {'n_theta':>8} | {'n_x':>8} | "
          f"{'theta_mean':>40} | {'x_mean':>8}")
    print("-" * 90)

    all_round_thetas = []
    all_round_xs = []
    for r in range(N_ROUNDS):
        try:
            r_theta = asnpe.get_round_thetas(r, seed=42)
            r_x = asnpe.get_round_xs(r, seed=42)
            all_round_thetas.append(r_theta)
            all_round_xs.append(r_x)

            r_theta_np = r_theta.numpy() if isinstance(r_theta, torch.Tensor) else np.array(r_theta)
            theta_means = [f"{r_theta_np[:, i].mean():.3f}" for i in range(4)]
            r_x_np = r_x.numpy() if isinstance(r_x, torch.Tensor) else np.array(r_x)

            print(f"{r:5d} | {r_theta_np.shape[0]:8d} | {r_x_np.shape[0]:8d} | "
                  f"[{', '.join(theta_means)}] | {r_x_np.mean():8.4f}")
        except Exception as e:
            print(f"{r:5d} | Error: {e}")

    # --------------------------------------------------------
    # DIAGNOSTIC PLOTS
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("  DIAGNOSTICS")
    print("=" * 65)

    # 1. Pairplot
    print("[1/6] Pairplot...")
    fig, axes = pairplot(
        samples_np,
        labels=PARAM_NAMES,
        limits=list(zip(PRIOR_LOW.numpy(), PRIOR_HIGH.numpy())),
        figsize=(10, 10),
    )
    fig.suptitle("ASNPE Posterior (TomTom x_obs, 11 features)", fontsize=14, y=1.02)
    plt.savefig("asnpe_pairplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: asnpe_pairplot.png")

    # 2. Marginals
    print("[2/6] Marginals...")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for i, (name, ax) in enumerate(zip(PARAM_NAMES, axes)):
        col = samples_np[:, i]
        lo, hi = PRIOR_LOW[i].item(), PRIOR_HIGH[i].item()

        ax.hist(col, bins=80, density=True, alpha=0.7, color="darkorange",
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

    plt.suptitle("ASNPE Marginal Posteriors — TomTom x_obs", fontsize=13)
    plt.tight_layout()
    plt.savefig("asnpe_marginals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: asnpe_marginals.png")

    # 3. Round-wise theta evolution
    print("[3/6] Round-wise evolution...")
    if len(all_round_thetas) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, (name, ax) in enumerate(zip(PARAM_NAMES, axes)):
            lo, hi = PRIOR_LOW[i].item(), PRIOR_HIGH[i].item()

            for r, r_theta in enumerate(all_round_thetas):
                r_np = r_theta.numpy() if isinstance(r_theta, torch.Tensor) else np.array(r_theta)
                color = plt.cm.viridis(r / max(len(all_round_thetas) - 1, 1))
                ax.hist(r_np[:, i], bins=30, density=True, alpha=0.35,
                        color=color, label=f"R{r}" if r % 2 == 0 else None)

            # Add final posterior
            ax.hist(samples_np[:, i], bins=50, density=True, alpha=0.3,
                    color="red", edgecolor="red", linewidth=0.5, label="Final posterior")

            ax.set_xlabel(name)
            ax.set_ylabel("Density")
            ax.set_xlim(lo, hi)
            ax.set_title(f"{name} — proposal focusing")
            ax.legend(fontsize=6, loc="upper right")

        plt.suptitle("ASNPE: How proposals concentrate over rounds", fontsize=14)
        plt.tight_layout()
        plt.savefig("asnpe_round_evolution.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: asnpe_round_evolution.png")

    # 4. Per-round parameter means
    print("[4/6] Parameter convergence...")
    if len(all_round_thetas) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        axes = axes.flatten()

        for i, (name, ax) in enumerate(zip(PARAM_NAMES, axes)):
            means = []
            stds = []
            for r_theta in all_round_thetas:
                r_np = r_theta.numpy() if isinstance(r_theta, torch.Tensor) else np.array(r_theta)
                means.append(r_np[:, i].mean())
                stds.append(r_np[:, i].std())

            rounds = range(len(means))
            ax.plot(rounds, means, "o-", color="darkorange", markersize=6, linewidth=2)
            ax.fill_between(rounds,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.2, color="darkorange")

            # SNPE result for comparison
            snpe_results = dict(np.load("snpe_results.npz", allow_pickle=True))
            snpe_mean = float(snpe_results["mean"][i])
            ax.axhline(snpe_mean, color="steelblue", linestyle="--",
                       linewidth=1.5, label=f"SNPE mean={snpe_mean:.3f}")

            ax.set_xlabel("Round")
            ax.set_ylabel(name)
            ax.set_title(f"{name} convergence")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        plt.suptitle("ASNPE Parameter Convergence (orange) vs SNPE (blue dashed)", fontsize=14)
        plt.tight_layout()
        plt.savefig("asnpe_convergence.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: asnpe_convergence.png")

    # 5. Budget usage
    print("[5/6] Budget usage...")
    if len(all_round_thetas) > 0:
        round_sizes = [
            (t.numpy() if isinstance(t, torch.Tensor) else np.array(t)).shape[0]
            for t in all_round_thetas
        ]
        cumulative = np.cumsum(round_sizes)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.bar(range(len(round_sizes)), round_sizes, color="darkorange", alpha=0.7)
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Simulations")
        ax1.set_title("Sims per round")

        ax2.plot(range(len(cumulative)), cumulative, "o-", color="darkorange",
                 markersize=6, linewidth=2)
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Cumulative simulations")
        ax2.set_title("Cumulative budget")
        ax2.axhline(N_ROUNDS * N_SAMPLES_PER_ROUND, color="gray",
                     linestyle="--", alpha=0.5, label="Budget")
        ax2.legend()

        plt.tight_layout()
        plt.savefig("asnpe_budget.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: asnpe_budget.png")

    # 6. seqinf built-in diagnostics
    print("[6/6] seqinf diagnostics...")
    try:
        diag = BayesianInferenceDiagnostic(asnpe, num_workers=1)

        try:
            entropies = diag.component_entropies(seed=42)
            print(f"  Component entropies: {entropies}")
        except Exception as e:
            print(f"  Component entropies: {e}")

        try:
            pp_ent = diag.pp_entropy(seed=42)
            print(f"  PP entropy: {pp_ent}")
        except Exception as e:
            print(f"  PP entropy: {e}")

        for plot_name, plot_func in [
            ("asnpe_diag_thetas.png", lambda: diag.plot_run_thetas(seed=42)),
            ("asnpe_diag_proposals.png", lambda: diag.plot_run_proposals(seed=42)),
        ]:
            try:
                plot_func()
                plt.savefig(plot_name, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  Saved: {plot_name}")
            except Exception as e:
                print(f"  {plot_name}: {e}")

    except Exception as e:
        print(f"  seqinf diagnostics failed: {e}")

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("  SAVING")
    print("=" * 65)

    with open("asnpe_posterior.pkl", "wb") as f:
        pickle.dump(posterior, f)
    print("  asnpe_posterior.pkl")

    try:
        with open("asnpe_inference.pkl", "wb") as f:
            pickle.dump(asnpe, f)
        print("  asnpe_inference.pkl")
    except Exception as e:
        print(f"  asnpe_inference.pkl FAILED: {e}")

    torch.save(torch.tensor(samples_np, dtype=torch.float32), "asnpe_samples.pt")
    print("  asnpe_samples.pt")

    results = {
        "param_names": PARAM_NAMES,
        "map": np.array(map_estimates),
        "mean": samples_np.mean(axis=0),
        "median": np.median(samples_np, axis=0),
        "std": samples_np.std(axis=0),
        "q05": np.percentile(samples_np, 5, axis=0),
        "q95": np.percentile(samples_np, 95, axis=0),
        "n_rounds": N_ROUNDS,
        "n_per_round": N_SAMPLES_PER_ROUND,
        "total_sims": _sim_counter["count"],
        "total_failures": _sim_counter["fail"],
        "total_time_s": total_time,
        "n_features": len(KEPT_INDICES),
    }
    np.savez("asnpe_results.npz", **results)
    print("  asnpe_results.npz")

    if len(all_round_thetas) > 0:
        torch.save(
            {
                "thetas": [t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
                           for t in all_round_thetas],
                "xs": [x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
                       for x in all_round_xs],
            },
            "asnpe_round_data.pt",
        )
        print("  asnpe_round_data.pt")

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("  ASNPE RESULTS SUMMARY")
    print("=" * 65)
    print(f"\n  Runtime:     {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    print(f"  Rounds:      {N_ROUNDS}")
    print(f"  Total sims:  {_sim_counter['count']} (failed: {_sim_counter['fail']})")
    print(f"  Features:    {len(KEPT_INDICES)}")
    print(f"  Sampling:    {N_POSTERIOR_SAMPLES} samples in {sample_time:.1f}s")

    print(f"\n  Inferred parameters (TomTom x_obs):")
    print(f"  {'Parameter':>15} | {'MAP':>8} | {'Mean±Std':>15} | {'90% CI':>18}")
    print(f"  " + "-" * 60)
    for i, name in enumerate(PARAM_NAMES):
        col = samples_np[:, i]
        q05, q95 = np.percentile(col, [5, 95])
        print(f"  {name:>15} | {map_estimates[i]:8.4f} | "
              f"{col.mean():7.4f} ± {col.std():.4f} | [{q05:.4f}, {q95:.4f}]")

    # Quick comparison with SNPE
    try:
        snpe_r = dict(np.load("snpe_results.npz", allow_pickle=True))
        print(f"\n  Quick comparison with SNPE:")
        print(f"  {'Parameter':>15} | {'SNPE Mean':>10} {'SNPE Std':>10} | "
              f"{'ASNPE Mean':>10} {'ASNPE Std':>10} | {'Sharper?':>8}")
        print(f"  " + "-" * 75)
        for i, name in enumerate(PARAM_NAMES):
            snpe_mean = float(snpe_r["mean"][i])
            snpe_std = float(snpe_r["std"][i])
            asnpe_mean = samples_np[:, i].mean()
            asnpe_std = samples_np[:, i].std()
            sharper = "ASNPE" if asnpe_std < snpe_std else "SNPE"
            print(f"  {name:>15} | {snpe_mean:10.4f} {snpe_std:10.4f} | "
                  f"{asnpe_mean:10.4f} {asnpe_std:10.4f} | {sharper:>8}")
    except Exception:
        pass

    print(f"\n  Next: python compare_methods.py")


if __name__ == "__main__":
    main()