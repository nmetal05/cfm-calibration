"""
Compare SNPE vs ASNPE posteriors.

Usage: python compare_methods.py
Input:  snpe_samples.pt, asnpe_samples.pt, snpe_results.npz, asnpe_results.npz
Output: comparison plots + summary table
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, wasserstein_distance

PARAM_NAMES = ["speedFactor", "speedDev", "sigma", "tau"]
PRIOR_LOW = [0.5, 0.0, 0.0, 0.5]
PRIOR_HIGH = [1.3, 0.3, 1.0, 3.0]


def load_results(prefix):
    """Load samples and results for a method."""
    samples = torch.load(f"{prefix}_samples.pt", weights_only=True).numpy()
    results = dict(np.load(f"{prefix}_results.npz", allow_pickle=True))
    return samples, results


def compute_sharpness(samples, prior_low, prior_high):
    """
    Ratio of posterior std to prior std.
    Lower = sharper = more informative.
    """
    prior_std = (np.array(prior_high) - np.array(prior_low)) / np.sqrt(12)
    post_std = samples.std(axis=0)
    return post_std / prior_std


def main():
    print("=" * 70)
    print("  SNPE vs ASNPE COMPARISON")
    print("=" * 70)

    # ---- Load ----
    methods = {}
    colors = {}

    try:
        snpe_samples, snpe_results = load_results("snpe")
        methods["SNPE"] = {"samples": snpe_samples, "results": snpe_results}
        colors["SNPE"] = "steelblue"
        print(f"  SNPE:  {snpe_samples.shape[0]} samples loaded")
    except FileNotFoundError:
        print("  SNPE:  NOT FOUND — skipping")

    try:
        asnpe_samples, asnpe_results = load_results("asnpe")
        methods["ASNPE"] = {"samples": asnpe_samples, "results": asnpe_results}
        colors["ASNPE"] = "darkorange"
        print(f"  ASNPE: {asnpe_samples.shape[0]} samples loaded")
    except FileNotFoundError:
        print("  ASNPE: NOT FOUND — skipping")

    if len(methods) < 2:
        print("\nNeed both SNPE and ASNPE results to compare.")
        print("Run train_snpe.py and train_asnpe.py first.")
        return

    # ---- Summary Table ----
    print("\n" + "=" * 70)
    print("  PARAMETER ESTIMATES")
    print("=" * 70)

    header = f"{'Parameter':>15}"
    for name in methods:
        header += f" | {name + ' MAP':>10} {name + ' Mean':>10} {name + ' Std':>8} {name + ' 90%CI':>18}"
    print(header)
    print("-" * len(header))

    for i, pname in enumerate(PARAM_NAMES):
        row = f"{pname:>15}"
        for mname, mdata in methods.items():
            s = mdata["samples"][:, i]
            q05, q95 = np.percentile(s, [5, 95])
            try:
                kde = gaussian_kde(s)
                grid = np.linspace(PRIOR_LOW[i], PRIOR_HIGH[i], 1000)
                map_val = grid[kde(grid).argmax()]
            except Exception:
                map_val = np.median(s)
            row += (f" | {map_val:10.4f} {s.mean():10.4f} {s.std():8.4f} "
                    f"[{q05:.4f},{q95:.4f}]")
        print(row)

    # ---- Sharpness comparison ----
    print("\n" + "=" * 70)
    print("  SHARPNESS (posterior_std / prior_std, lower = sharper)")
    print("=" * 70)
    print(f"{'Parameter':>15}", end="")
    for mname in methods:
        print(f" | {mname:>12}", end="")
    print(f" | {'Winner':>12}")
    print("-" * 65)

    sharpness = {}
    for mname, mdata in methods.items():
        sharpness[mname] = compute_sharpness(mdata["samples"], PRIOR_LOW, PRIOR_HIGH)

    for i, pname in enumerate(PARAM_NAMES):
        print(f"{pname:>15}", end="")
        vals = {}
        for mname in methods:
            val = sharpness[mname][i]
            vals[mname] = val
            print(f" | {val:12.4f}", end="")
        winner = min(vals, key=vals.get)
        print(f" | {winner:>12}")

    # Overall sharpness
    print(f"{'MEAN':>15}", end="")
    mean_vals = {}
    for mname in methods:
        mv = sharpness[mname].mean()
        mean_vals[mname] = mv
        print(f" | {mv:12.4f}", end="")
    winner = min(mean_vals, key=mean_vals.get)
    print(f" | {winner:>12}")

    # ---- Wasserstein distance between posteriors ----
    print("\n" + "=" * 70)
    print("  WASSERSTEIN DISTANCE BETWEEN POSTERIORS")
    print("=" * 70)
    method_names = list(methods.keys())
    for i, pname in enumerate(PARAM_NAMES):
        s1 = methods[method_names[0]]["samples"][:, i]
        s2 = methods[method_names[1]]["samples"][:, i]
        wd = wasserstein_distance(s1, s2)
        prior_range = PRIOR_HIGH[i] - PRIOR_LOW[i]
        wd_norm = wd / prior_range
        print(f"  {pname:>15}: W={wd:.4f} (normalized: {wd_norm:.4f})")

    # ---- Runtime comparison ----
    print("\n" + "=" * 70)
    print("  COMPUTATIONAL COST")
    print("=" * 70)
    for mname, mdata in methods.items():
        r = mdata["results"]
        total_time = float(r.get("total_time_s", r.get("train_time_s", 0)))
        n_sims = int(r.get("total_sims", r.get("n_sims", 0)))
        print(f"  {mname:>8}: {total_time/60:.1f} min | {n_sims} sims | "
              f"{n_sims/(total_time/60):.0f} sims/min" if total_time > 0 else
              f"  {mname:>8}: unknown time | {n_sims} sims")

    # ============================================================
    # PLOTS
    # ============================================================
    print("\n--- Generating Comparison Plots ---")

    # 1. Overlaid marginals
    print("[1/4] Overlaid marginal posteriors...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, (pname, ax) in enumerate(zip(PARAM_NAMES, axes)):
        lo, hi = PRIOR_LOW[i], PRIOR_HIGH[i]

        # Prior
        ax.axhline(1.0 / (hi - lo), color="gray", linestyle="--",
                    alpha=0.5, linewidth=1.5, label="Prior")

        for mname, mdata in methods.items():
            col = mdata["samples"][:, i]
            ax.hist(col, bins=80, density=True, alpha=0.4,
                    color=colors[mname], edgecolor="white", linewidth=0.3,
                    label=mname)

            # KDE overlay
            try:
                kde = gaussian_kde(col)
                grid = np.linspace(lo, hi, 300)
                ax.plot(grid, kde(grid), color=colors[mname], linewidth=2)
            except Exception:
                pass

        ax.set_xlabel(pname, fontsize=12)
        ax.set_ylabel("Density" if i == 0 else "")
        ax.set_xlim(lo, hi)
        ax.set_title(pname, fontsize=12)
        ax.legend(fontsize=9)

    plt.suptitle("SNPE (blue) vs ASNPE (orange) — Marginal Posteriors", fontsize=14)
    plt.tight_layout()
    plt.savefig("compare_marginals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: compare_marginals.png")

    # 2. Side-by-side pairplots
    print("[2/4] Side-by-side pairplots...")
    fig, all_axes = plt.subplots(4, 8, figsize=(24, 12))

    for col_offset, (mname, mdata) in enumerate(methods.items()):
        for i in range(4):
            for j in range(4):
                ax = all_axes[i, j + col_offset * 4]

                if i == j:
                    # Diagonal: histogram
                    s = mdata["samples"][:, i]
                    ax.hist(s, bins=40, density=True, alpha=0.7,
                            color=colors[mname], edgecolor="white", linewidth=0.3)
                    ax.set_xlim(PRIOR_LOW[i], PRIOR_HIGH[i])
                    if i == 0:
                        ax.set_title(mname, fontsize=12, fontweight="bold")
                elif i > j:
                    # Lower triangle: scatter
                    ax.scatter(
                        mdata["samples"][::10, j],
                        mdata["samples"][::10, i],
                        alpha=0.1, s=1, color=colors[mname],
                    )
                    ax.set_xlim(PRIOR_LOW[j], PRIOR_HIGH[j])
                    ax.set_ylim(PRIOR_LOW[i], PRIOR_HIGH[i])
                else:
                    ax.axis("off")

                # Labels
                if i == 3:
                    ax.set_xlabel(PARAM_NAMES[j], fontsize=8)
                if j == 0 + col_offset * 4:
                    ax.set_ylabel(PARAM_NAMES[i], fontsize=8)
                ax.tick_params(labelsize=6)

    plt.suptitle("SNPE (left) vs ASNPE (right) — Pairplots", fontsize=14)
    plt.tight_layout()
    plt.savefig("compare_pairplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: compare_pairplots.png")

    # 3. Sharpness bar chart
    print("[3/4] Sharpness comparison...")
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(PARAM_NAMES))
    width = 0.35

    for k, (mname, mdata) in enumerate(methods.items()):
        sharp = sharpness[mname]
        ax.bar(x_pos + k * width, sharp, width, label=mname,
               color=colors[mname], alpha=0.8, edgecolor="white")

    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels(PARAM_NAMES, fontsize=11)
    ax.set_ylabel("Posterior Std / Prior Std", fontsize=11)
    ax.set_title("Sharpness Comparison (lower = more informative)", fontsize=13)
    ax.legend(fontsize=11)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Prior level")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("compare_sharpness.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: compare_sharpness.png")

    # 4. Summary card
    print("[4/4] Summary card...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    summary_text = "SNPE vs ASNPE — Summary\n"
    summary_text += "=" * 50 + "\n\n"

    for i, pname in enumerate(PARAM_NAMES):
        summary_text += f"{pname}:\n"
        for mname, mdata in methods.items():
            s = mdata["samples"][:, i]
            q05, q95 = np.percentile(s, [5, 95])
            summary_text += (f"  {mname:>6}: {s.mean():.4f} +/- {s.std():.4f}  "
                             f"90%CI=[{q05:.4f}, {q95:.4f}]\n")
        summary_text += "\n"

    summary_text += f"Sharpness (mean posterior_std/prior_std):\n"
    for mname in methods:
        summary_text += f"  {mname:>6}: {sharpness[mname].mean():.4f}\n"

    summary_text += f"\nOverall winner: "
    overall_winner = min(mean_vals, key=mean_vals.get)
    summary_text += f"{overall_winner} (sharper posteriors)\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.savefig("compare_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: compare_summary.png")

    # ---- Final ----
    print("\n" + "=" * 70)
    print("  COMPARISON COMPLETE")
    print("=" * 70)
    print(f"\n  Overall winner (sharpness): {overall_winner}")
    print(f"\n  Files:")
    print(f"    compare_marginals.png   — overlaid marginal posteriors")
    print(f"    compare_pairplots.png   — side-by-side joint posteriors")
    print(f"    compare_sharpness.png   — sharpness bar chart")
    print(f"    compare_summary.png     — text summary card")


if __name__ == "__main__":
    main()