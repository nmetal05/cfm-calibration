"""
Compute RMSE between model predictions and real TomTom data
This tells us which model best matches the observed data
"""

import numpy as np
import torch
from pathlib import Path
import subprocess
import sys

# Add Krauss folder to path for imports
sys.path.insert(0, str(Path(r"C:\Users\PC\Downloads\Krauss")))
from write_vtype import write_vtype_file
from parse_output_shared import (
    parse_edge_data,
    summary_statistics_sumo,
    get_edge_max_speeds,
)

# Paths to each model
MODELS = {
    "Krauss": Path(r"C:\Users\PC\Downloads\Krauss"),
    "Wiedemann": Path(r"C:\Users\PC\Downloads\Wiedemann\Wiedemann"),
    "IDM": Path(r"C:\Users\PC\Downloads\IDM\IDM"),
}


def run_sumo_simulation(model_dir, theta, sumo_binary="sumo"):
    """Run SUMO with given parameters and return summary stats"""
    sim_dir = model_dir / "rmse_check"
    sim_dir.mkdir(exist_ok=True)

    vtype_path = sim_dir / "vtype.xml"
    write_vtype_file(theta.tolist(), str(vtype_path))

    edgedata_output = sim_dir / "edgedata.xml"
    edgedata_add = sim_dir / "edgedata.add.xml"

    with open(edgedata_add, "w") as f:
        f.write(
            "<additional>\n"
            f'    <edgeData id="rmse" freq="300" '
            f'file="{edgedata_output.resolve()}" excludeEmpty="true"/>\n'
            "</additional>\n"
        )

    try:
        result = subprocess.run(
            [
                sumo_binary,
                "-c",
                str(model_dir / "sbi_peak.sumocfg"),
                "--additional-files",
                f"{vtype_path.resolve()},{edgedata_add.resolve()}",
                "--seed",
                "42",
            ],
            capture_output=True,
            timeout=120,
            cwd=str(model_dir.resolve()),
        )

        if result.returncode == 0 and edgedata_output.exists():
            edge_data = parse_edge_data(str(edgedata_output))
            edge_speeds, edge_lengths = get_edge_max_speeds(model_dir / "osm.net.xml")
            x = summary_statistics_sumo(edge_data, edge_speeds, edge_lengths)

            for f in sim_dir.glob("*"):
                f.unlink(missing_ok=True)
            sim_dir.rmdir()

            return x
    except:
        pass

    for f in sim_dir.glob("*"):
        f.unlink(missing_ok=True)
    return None


def compute_rmse(pred, obs):
    return np.sqrt(np.mean((pred - obs) ** 2))


def compute_mae(pred, obs):
    return np.mean(np.abs(pred - obs))


print("=" * 70)
print(" " * 20 + "MODEL FIT COMPARISON (RMSE)")
print("=" * 70)

# Load observed data
for model_name, model_dir in MODELS.items():
    if (model_dir / "x_obs.pt").exists():
        x_obs = torch.load(model_dir / "x_obs.pt", weights_only=True).numpy()
        break

print(f"\nObserved data shape: {x_obs.shape}")

# Load feature selection
feat_sel = torch.load(
    MODELS["Krauss"] / "snpe_feature_selection.pt", weights_only=False
)
kept_indices = feat_sel["kept_original_indices"]
feat_names = feat_sel["final_names"]

print(f"Using {len(kept_indices)} features: {feat_names}")

x_obs_kept = x_obs[kept_indices]
print(f"x_obs kept shape: {x_obs_kept.shape}")

# Load MAP estimates
results = {}
for model_name, model_dir in MODELS.items():
    r = np.load(model_dir / "snpe_results.npz", allow_pickle=True)
    results[model_name] = {
        "map": r["map"],
        "mean": r["mean"],
    }

print("\n" + "=" * 70)
print("Computing RMSE for each model using MAP estimates...")
print("=" * 70)

test_params = {
    "Krauss (MAP)": results["Krauss"]["map"],
    "Wiedemann (MAP)": results["Wiedemann"]["map"],
    "IDM (MAP)": results["IDM"]["map"],
    "Krauss (Mean)": results["Krauss"]["mean"],
    "Wiedemann (Mean)": results["Wiedemann"]["mean"],
    "IDM (Mean)": results["IDM"]["mean"],
}

test_dir = MODELS["Krauss"]

rmse_results = {}
mae_results = {}

for name, params in test_params.items():
    print(f"\nRunning {name}...")
    print(f"  Params: {params}")

    x_sim = run_sumo_simulation(test_dir, torch.tensor(params))

    if x_sim is not None:
        x_sim_kept = x_sim[kept_indices]

        rmse = compute_rmse(x_sim_kept, x_obs_kept)
        mae = compute_mae(x_sim_kept, x_obs_kept)

        rmse_results[name] = rmse
        mae_results[name] = mae

        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")

        print(f"  Top 3 feature differences:")
        diffs = np.abs(x_sim_kept - x_obs_kept)
        top_idx = np.argsort(diffs)[::-1][:3]
        for idx in top_idx:
            fname = feat_names[kept_indices.index(kept_indices[idx])]
            print(
                f"    {fname}: sim={x_sim_kept[idx]:.3f}, obs={x_obs_kept[idx]:.3f}, diff={diffs[idx]:.3f}"
            )
    else:
        print(f"  FAILED to run simulation")
        rmse_results[name] = float("inf")
        mae_results[name] = float("inf")

print("\n" + "=" * 70)
print(" " * 25 + "RMSE RESULTS SUMMARY")
print("=" * 70)

sorted_results = sorted(rmse_results.items(), key=lambda x: x[1])

print(f"\n{'Model':<25} | {'RMSE':>10} | {'MAE':>10} | {'Rank':>6}")
print("-" * 55)
for rank, (name, rmse) in enumerate(sorted_results, 1):
    mae = mae_results[name]
    print(f"{name:<25} | {rmse:>10.4f} | {mae:>10.4f} | {rank:>6}")

best_model = sorted_results[0][0]
print(f"\n*** BEST MODEL: {best_model} (lowest RMSE) ***")

print("\n" + "=" * 70)
print(" " * 25 + "INTERPRETATION")
print("=" * 70)
print("""
RMSE (Root Mean Square Error):
- Measures the average magnitude of prediction errors
- Lower = better fit to real TomTom data
- Penalizes large errors more heavily

MAE (Mean Absolute Error):
- Average absolute difference
- Lower = better fit
- More robust to outliers
""")
