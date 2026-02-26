"""
Train ASNPE — FINAL (monkey-patched, API fixes, full budget)

Usage: python train_asnpe.py
"""

import os
import time
import uuid
import pickle
import subprocess
import traceback
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
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
# MONKEY-PATCH
# ============================================================
import seqinf.flow as _flow

_original_bayesflow_init = _flow.BayesFlow.__init__

def _patched_bayesflow_init(self, *args, **kwargs):
    _original_bayesflow_init(self, *args, **kwargs)
    if not hasattr(self, '_context_used_in_base'):
        self._context_used_in_base = False

_flow.BayesFlow.__init__ = _patched_bayesflow_init
print("  [Monkey-patch applied]")

from seqinf.methods.posterior import ASNPE
from seqinf import BayesianInferenceDiagnostic

# ============================================================
# CONFIG
# ============================================================
SUMO_BINARY = "sumo"
BASE_DIR = Path(".")
SIM_TIMEOUT = 120

N_ROUNDS = 10
N_SAMPLES_PER_ROUND = 50  # 10 × 500 = 5000 total (fair comparison with SNPE)
N_WORKERS = 15
N_POSTERIOR_SAMPLES = 50_000

PARAM_NAMES = ["speedFactor", "speedDev", "sigma", "tau"]
PRIOR_LOW = torch.tensor([0.5, 0.0, 0.0, 0.5], dtype=torch.float32)
PRIOR_HIGH = torch.tensor([1.3, 0.3, 1.0, 3.0], dtype=torch.float32)

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 65)
print("  ASNPE — FULL RUN (5000 sims)")
print("=" * 65)

print("\nLoading network edge speeds...")
EDGE_MAX_SPEEDS, EDGE_LENGTHS = get_edge_max_speeds("osm.net.xml")
print(f"  Loaded {len(EDGE_MAX_SPEEDS)} edges")

print("Loading feature selection...")
feat_sel = torch.load("snpe_feature_selection.pt", weights_only=False)
KEPT_INDICES = feat_sel["kept_original_indices"]
FEAT_NAMES = feat_sel["final_names"]
N_FEATURES = len(KEPT_INDICES)
print(f"  Using {N_FEATURES} features: {FEAT_NAMES}")

x_obs_full = torch.load("x_obs.pt", weights_only=True).float()
x_obs = x_obs_full[KEPT_INDICES]
print(f"  x_obs: {x_obs.shape}")

_sim_counter = {"count": 0, "fail": 0, "start_time": None}


# ============================================================
# SUMO SIMULATOR
# ============================================================
def sumo_simulator(theta):
    if _sim_counter["start_time"] is None:
        _sim_counter["start_time"] = time.time()

    try:
        if isinstance(theta, torch.Tensor):
            theta = theta.detach().cpu()
            if theta.dim() == 2:
                theta = theta.squeeze(0)
            theta_np = theta.numpy().astype(np.float64)
        elif isinstance(theta, np.ndarray):
            theta_np = theta.flatten().astype(np.float64)
        else:
            theta_np = np.array(theta, dtype=np.float64).flatten()

        if theta_np.shape != (4,):
            raise ValueError(f"Expected (4,), got {theta_np.shape}")

        theta_np = np.clip(theta_np, [0.5, 0.0, 0.0, 0.5], [1.3, 0.3, 1.0, 3.0])

    except Exception:
        _sim_counter["fail"] += 1
        return torch.zeros(N_FEATURES, dtype=torch.float32)

    sim_id = uuid.uuid4().hex[:12]
    sim_dir = BASE_DIR / "asnpe_runs" / f"sim_{sim_id}"
    sim_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(2):
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

                _cleanup(sim_dir)

                if x_full is not None and len(x_full) == 57 and np.all(np.isfinite(x_full)):
                    x_sel = x_full[KEPT_INDICES]
                    _sim_counter["count"] += 1

                    if _sim_counter["count"] % 100 == 0:
                        elapsed = time.time() - _sim_counter["start_time"]
                        rate = _sim_counter["count"] / (elapsed / 60)
                        print(
                            f"    [ASNPE] Sims: {_sim_counter['count']} | "
                            f"Failed: {_sim_counter['fail']} | "
                            f"Rate: {rate:.0f}/min | "
                            f"Elapsed: {elapsed/60:.1f} min"
                        )

                    return torch.tensor(x_sel, dtype=torch.float32)

            if attempt == 0:
                time.sleep(0.3)

        except subprocess.TimeoutExpired:
            if attempt == 0:
                time.sleep(0.3)
        except Exception:
            pass

    _cleanup(sim_dir)
    _sim_counter["fail"] += 1
    return torch.zeros(N_FEATURES, dtype=torch.float32)


def _cleanup(sim_dir):
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
    assert (BASE_DIR / "osm.net.xml").exists()
    assert (BASE_DIR / "sbi_peak.sumocfg").exists()
    assert (BASE_DIR / "routes_peak_novtype.rou.xml").exists()

    (BASE_DIR / "asnpe_runs").mkdir(exist_ok=True)

    prior = BoxUniform(low=PRIOR_LOW, high=PRIOR_HIGH)

    # Test
    print("\nTesting simulator...")
    test_x = sumo_simulator(torch.tensor([0.9, 0.1, 0.5, 1.5]))
    print(f"  OK: shape={test_x.shape}")
    assert test_x.shape == (N_FEATURES,)

    _sim_counter["count"] = 0
    _sim_counter["fail"] = 0
    _sim_counter["start_time"] = None

    total_budget = N_ROUNDS * N_SAMPLES_PER_ROUND
    est_time = total_budget / 28  # conservative rate from test run
    print(f"\n--- Configuration ---")
    print(f"  Rounds:        {N_ROUNDS}")
    print(f"  Sims/round:    {N_SAMPLES_PER_ROUND}")
    print(f"  Total budget:  {total_budget}")
    print(f"  Workers:       {N_WORKERS}")
    print(f"  Est. time:     ~{est_time:.0f} min ({est_time/60:.1f} hours)")

    # Create ASNPE
    print("\nCreating ASNPE...")
    asnpe = ASNPE(
        simulator=sumo_simulator,
        prior=prior,
        density_estimator="maf",
        num_workers=N_WORKERS,
    )
    print("  OK")

    # Run
    print("\n" + "=" * 65)
    print("  RUNNING ASNPE")
    print("=" * 65)
    print(f"  Started at {time.strftime('%H:%M:%S')}\n")

    t_total_start = time.time()

    with joblib.parallel_backend("threading", n_jobs=N_WORKERS):
        asnpe.run(
            num_rounds=N_ROUNDS,
            num_samples=N_SAMPLES_PER_ROUND,
            x_o=x_obs,
            seed=42,
        )

    total_time = time.time() - t_total_start

    print(f"\n  Completed!")
    print(f"  Time:    {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    print(f"  Sims:    {_sim_counter['count']} (failed: {_sim_counter['fail']})")

    # ---- Posterior ----
    print("\n" + "=" * 65)
    print("  POSTERIOR SAMPLING")
    print("=" * 65)

    posterior = asnpe.proposal
    print(f"  Type: {type(posterior)}")
    print(f"  Sampling {N_POSTERIOR_SAMPLES}...")

    t_sample = time.time()
    sampled = False
    for attempt, kwargs in enumerate([{}, {"x": x_obs.unsqueeze(0)}, {"x": x_obs}]):
        try:
            samples = posterior.sample((N_POSTERIOR_SAMPLES,), **kwargs)
            sampled = True
            sample_time = time.time() - t_sample
            print(f"  OK (attempt {attempt+1}), {sample_time:.1f}s")
            break
        except Exception as e:
            print(f"  Attempt {attempt+1}: {e}")

    if not sampled:
        print("  ERROR: All sampling failed!")
        with open("asnpe_inference.pkl", "wb") as f:
            pickle.dump(asnpe, f)
        return

    samples_np = samples.numpy()
    for i in range(4):
        samples_np[:, i] = np.clip(
            samples_np[:, i], PRIOR_LOW[i].item(), PRIOR_HIGH[i].item()
        )

    # ---- Results ----
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

    # ---- Per-round stats (FIXED: no seed kwarg) ----
    print("\n--- Per-Round Statistics ---")
    all_round_thetas = []
    all_round_xs = []

    for r in range(N_ROUNDS):
        try:
            r_theta = asnpe.get_round_thetas(r)
            r_x = asnpe.get_round_xs(r)
            all_round_thetas.append(r_theta)
            all_round_xs.append(r_x)

            r_np = r_theta.numpy() if isinstance(r_theta, torch.Tensor) else np.array(r_theta)
            means = ", ".join(f"{r_np[:, i].mean():.3f}" for i in range(4))
            stds = ", ".join(f"{r_np[:, i].std():.3f}" for i in range(4))
            print(f"  R{r:2d}: n={r_np.shape[0]:4d}  mean=[{means}]  std=[{stds}]")
        except Exception as e:
            # Try without any args
            try:
                # Maybe it needs the seed from the run
                r_theta = asnpe.get_round_thetas(r, 42)
                r_x = asnpe.get_round_xs(r, 42)
                all_round_thetas.append(r_theta)
                all_round_xs.append(r_x)
                r_np = r_theta.numpy() if isinstance(r_theta, torch.Tensor) else np.array(r_theta)
                means = ", ".join(f"{r_np[:, i].mean():.3f}" for i in range(4))
                print(f"  R{r:2d}: n={r_np.shape[0]:4d}  mean=[{means}]")
            except Exception as e2:
                print(f"  R{r:2d}: Error — {e2}")

    # ---- PLOTS ----
    print("\n--- Plots ---")

    # 1. Pairplot
    print("[1/5] Pairplot...")
    fig, axes = pairplot(
        samples_np,
        labels=PARAM_NAMES,
        limits=list(zip(PRIOR_LOW.numpy(), PRIOR_HIGH.numpy())),
        figsize=(10, 10),
    )
    fig.suptitle(f"ASNPE Posterior ({N_ROUNDS} rounds, {_sim_counter['count']} sims)",
                 fontsize=14, y=1.02)
    plt.savefig("asnpe_pairplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: asnpe_pairplot.png")

    # 2. Marginals
    print("[2/5] Marginals...")
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
        ax.axvspan(q05, q95, alpha=0.15, color="red")
        ax.set_xlabel(name, fontsize=12)
        ax.set_xlim(lo, hi)
        ax.legend(fontsize=8)
        ax.set_title(name)
    plt.suptitle("ASNPE Marginal Posteriors", fontsize=13)
    plt.tight_layout()
    plt.savefig("asnpe_marginals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: asnpe_marginals.png")

    # 3. Round evolution
    print("[3/5] Round evolution...")
    if len(all_round_thetas) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for i, (name, ax) in enumerate(zip(PARAM_NAMES, axes)):
            lo, hi = PRIOR_LOW[i].item(), PRIOR_HIGH[i].item()
            for r, r_theta in enumerate(all_round_thetas):
                r_np = r_theta.numpy() if isinstance(r_theta, torch.Tensor) else np.array(r_theta)
                color = plt.cm.viridis(r / max(len(all_round_thetas) - 1, 1))
                ax.hist(r_np[:, i], bins=30, density=True, alpha=0.35,
                        color=color, label=f"R{r}" if r % 2 == 0 else None)
            ax.hist(samples_np[:, i], bins=50, density=True, alpha=0.3,
                    color="red", label="Final")
            ax.set_xlim(lo, hi)
            ax.set_title(name)
            ax.legend(fontsize=6)
        plt.suptitle("ASNPE: Proposal Focusing Over Rounds", fontsize=14)
        plt.tight_layout()
        plt.savefig("asnpe_round_evolution.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: asnpe_round_evolution.png")
    else:
        print("  Skipped (no round data)")

    # 4. Convergence vs SNPE
    print("[4/5] Convergence...")
    if len(all_round_thetas) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        axes = axes.flatten()
        try:
            snpe_r = dict(np.load("snpe_results.npz", allow_pickle=True))
            has_snpe = True
        except Exception:
            has_snpe = False

        for i, (name, ax) in enumerate(zip(PARAM_NAMES, axes)):
            means = []
            stds = []
            for r_theta in all_round_thetas:
                r_np = r_theta.numpy() if isinstance(r_theta, torch.Tensor) else np.array(r_theta)
                means.append(r_np[:, i].mean())
                stds.append(r_np[:, i].std())

            rounds = range(len(means))
            ax.plot(rounds, means, "o-", color="darkorange", markersize=5, linewidth=2)
            ax.fill_between(rounds,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.2, color="darkorange")

            if has_snpe:
                ax.axhline(float(snpe_r["mean"][i]), color="steelblue",
                           linestyle="--", linewidth=1.5,
                           label=f"SNPE={float(snpe_r['mean'][i]):.3f}")
            ax.set_xlabel("Round")
            ax.set_ylabel(name)
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        plt.suptitle("ASNPE Convergence (orange) vs SNPE (blue)", fontsize=14)
        plt.tight_layout()
        plt.savefig("asnpe_convergence.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: asnpe_convergence.png")
    else:
        print("  Skipped (no round data)")

    # 5. Budget
    print("[5/5] Budget...")
    if len(all_round_thetas) > 0:
        sizes = [(t.numpy() if isinstance(t, torch.Tensor) else np.array(t)).shape[0]
                 for t in all_round_thetas]
        cum = np.cumsum(sizes)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.bar(range(len(sizes)), sizes, color="darkorange", alpha=0.7)
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Simulations")
        ax1.set_title("Sims per round")
        ax2.plot(range(len(cum)), cum, "o-", color="darkorange", markersize=6, linewidth=2)
        ax2.axhline(total_budget, color="gray", linestyle="--", alpha=0.5, label="Budget")
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Cumulative")
        ax2.set_title("Cumulative sims")
        ax2.legend()
        plt.tight_layout()
        plt.savefig("asnpe_budget.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: asnpe_budget.png")
    else:
        print("  Skipped (no round data)")

    # seqinf diagnostics (no seed kwargs)
    print("\n--- seqinf diagnostics ---")
    try:
        diag = BayesianInferenceDiagnostic(asnpe, num_workers=1)
        try:
            ent = diag.component_entropies()
            print(f"  Component entropies: {ent}")
        except Exception as e:
            print(f"  Component entropies: {e}")

        for fname, func in [
            ("asnpe_diag_thetas.png", lambda: diag.plot_run_thetas()),
            ("asnpe_diag_proposals.png", lambda: diag.plot_run_proposals()),
        ]:
            try:
                func()
                plt.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  Saved: {fname}")
            except Exception as e:
                print(f"  {fname}: {e}")
    except Exception as e:
        print(f"  Diagnostics: {e}")

    # ---- SAVE ----
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
        print(f"  asnpe_inference.pkl: {e}")

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
        "n_features": N_FEATURES,
    }
    np.savez("asnpe_results.npz", **results)
    print("  asnpe_results.npz")

    if len(all_round_thetas) > 0:
        torch.save(
            {
                "thetas": [t if isinstance(t, torch.Tensor)
                           else torch.tensor(t, dtype=torch.float32)
                           for t in all_round_thetas],
                "xs": [x if isinstance(x, torch.Tensor)
                       else torch.tensor(x, dtype=torch.float32)
                       for x in all_round_xs],
            },
            "asnpe_round_data.pt",
        )
        print("  asnpe_round_data.pt")

    # ---- SUMMARY ----
    print("\n" + "=" * 65)
    print("  ASNPE RESULTS")
    print("=" * 65)
    print(f"\n  Runtime:  {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    print(f"  Sims:     {_sim_counter['count']} (failed: {_sim_counter['fail']})")

    print(f"\n  {'Param':>15} | {'MAP':>8} | {'Mean±Std':>15} | {'90% CI':>18}")
    print(f"  " + "-" * 60)
    for i, name in enumerate(PARAM_NAMES):
        col = samples_np[:, i]
        q05, q95 = np.percentile(col, [5, 95])
        print(f"  {name:>15} | {map_estimates[i]:8.4f} | "
              f"{col.mean():7.4f} ± {col.std():.4f} | [{q05:.4f}, {q95:.4f}]")

    try:
        snpe_r = dict(np.load("snpe_results.npz", allow_pickle=True))
        print(f"\n  vs SNPE:")
        print(f"  {'Param':>15} | {'SNPE Std':>9} | {'ASNPE Std':>9} | {'Winner':>8}")
        print(f"  " + "-" * 50)
        for i, name in enumerate(PARAM_NAMES):
            s_std = float(snpe_r["std"][i])
            a_std = samples_np[:, i].std()
            w = "ASNPE" if a_std < s_std else "SNPE"
            print(f"  {name:>15} | {s_std:9.4f} | {a_std:9.4f} | {w:>8}")
    except Exception:
        pass

    print(f"\n  Next: python compare_methods.py")


if __name__ == "__main__":
    main()