"""
Train ASNPE — v6 FULL (1000 sims, single run, fair SNPE comparison)

Budget: 10 rounds × 500 = 5,000 sims (same as SNPE)
Usage: python train_asnpe_v6.py
"""

import os
import sys
import time
import uuid
import pickle
import subprocess
import traceback
import threading
import importlib
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
# MONKEY-PATCH 1: Fix missing attribute
# ============================================================
import seqinf.flow as _flow

_original_bayesflow_init = _flow.BayesFlow.__init__


def _patched_bayesflow_init(self, *args, **kwargs):
    _original_bayesflow_init(self, *args, **kwargs)
    if not hasattr(self, "_context_used_in_base"):
        self._context_used_in_base = False


_flow.BayesFlow.__init__ = _patched_bayesflow_init
print("  [Patch 1 OK: BayesFlow._context_used_in_base]")

# ============================================================
# MONKEY-PATCH 2: Find and cap SNPE_C.train (brute force)
# ============================================================
MAX_TRAIN_EPOCHS = 300
TRAIN_BATCH_SIZE = 256

_training_log = {"rounds": []}
_patched_train_ok = False

import sbi


def _find_snpe_classes():
    found = []
    paths = [
        "sbi.inference.trainers.npe.npe_c",
        "sbi.inference.snpe.snpe_c",
        "sbi.inference.npe.npe_c",
        "sbi.inference",
        "sbi.inference.snpe",
        "sbi.inference.trainers",
        "sbi.inference.trainers.npe",
    ]
    for path in paths:
        try:
            mod = importlib.import_module(path)
            for attr_name in dir(mod):
                obj = getattr(mod, attr_name)
                if (
                    isinstance(obj, type)
                    and "SNPE" in attr_name
                    and hasattr(obj, "train")
                ):
                    found.append((f"{path}.{attr_name}", obj))
        except (ImportError, AttributeError):
            continue
    try:
        from sbi.inference import SNPE

        if hasattr(SNPE, "train"):
            found.append(("sbi.inference.SNPE", SNPE))
    except ImportError:
        pass
    return found


print("  [Patch 2: Searching for SNPE classes...]")
snpe_classes = _find_snpe_classes()
for name, cls in snpe_classes:
    print(f"    Found: {name}")

for name, cls in snpe_classes:
    try:
        _orig = cls.train

        def _make_patched(original, class_name):
            def _patched(self, *args, **kwargs):
                kwargs.setdefault("max_num_epochs", MAX_TRAIN_EPOCHS)
                if (
                    kwargs.get("max_num_epochs", MAX_TRAIN_EPOCHS + 1)
                    > MAX_TRAIN_EPOCHS
                ):
                    kwargs["max_num_epochs"] = MAX_TRAIN_EPOCHS
                kwargs.setdefault("training_batch_size", TRAIN_BATCH_SIZE)

                # Count cumulative data
                n_data = "?"
                try:
                    if hasattr(self, "_data_round_index"):
                        n_data = len(self._data_round_index)
                    elif hasattr(self, "_num_iid_data_points"):
                        n_data = self._num_iid_data_points
                except:
                    pass

                round_idx = len(_training_log["rounds"])
                print(f"\n    ┌─ NEURAL NETWORK TRAINING (Round {round_idx}) ───────")
                print(f"    │ Max epochs:   {kwargs['max_num_epochs']}")
                print(f"    │ Batch size:   {kwargs['training_batch_size']}")
                print(f"    │ Training data: {n_data} samples")
                print(f"    │ Training...", end="", flush=True)

                t0 = time.time()
                result = original(self, *args, **kwargs)
                dt = time.time() - t0

                metrics = {
                    "round": round_idx,
                    "train_time_s": dt,
                    "epochs": None,
                    "best_val_loss": None,
                    "final_train_loss": None,
                    "n_data": n_data,
                }

                try:
                    if hasattr(self, "_summary"):
                        s = self._summary
                        if "training_log_probs" in s:
                            metrics["epochs"] = len(s["training_log_probs"])
                            metrics["final_train_loss"] = float(
                                s["training_log_probs"][-1]
                            )
                            metrics["train_losses"] = [
                                float(x) for x in s["training_log_probs"]
                            ]
                        if "validation_log_probs" in s:
                            metrics["best_val_loss"] = float(
                                min(s["validation_log_probs"])
                            )
                            metrics["val_losses"] = [
                                float(x) for x in s["validation_log_probs"]
                            ]
                except:
                    pass

                print(f" done!")
                print(f"    │ Time:         {dt:.1f}s")
                if metrics["epochs"]:
                    hit_cap = (
                        " ⚠ HIT CAP" if metrics["epochs"] >= MAX_TRAIN_EPOCHS else ""
                    )
                    print(
                        f"    │ Epochs:       {metrics['epochs']}/{MAX_TRAIN_EPOCHS}{hit_cap}"
                    )
                if metrics["final_train_loss"] is not None:
                    print(f"    │ Train loss:   {metrics['final_train_loss']:.4f}")
                if metrics["best_val_loss"] is not None:
                    print(f"    │ Val loss:     {metrics['best_val_loss']:.4f}")
                print(f"    └──────────────────────────────────────────────")

                _training_log["rounds"].append(metrics)
                return result

            return _patched

        cls.train = _make_patched(_orig, name)
        _patched_train_ok = True
        print(f"    ✓ Patched: {name}")
    except Exception as e:
        print(f"    ✗ {name}: {e}")

if not _patched_train_ok:
    print("  [Patch 2: Trying seqinf internal fallback...]")
    try:
        import seqinf.methods.posterior as _sp

        _orig_init = _sp.ASNPE.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            for attr in ["_inference", "inference", "_snpe", "snpe"]:
                inf = getattr(self, attr, None)
                if inf and hasattr(inf, "train"):
                    _orig_t = inf.train

                    def _cap(orig=_orig_t):
                        def _inner(*a, **kw):
                            kw.setdefault("max_num_epochs", MAX_TRAIN_EPOCHS)
                            if kw.get("max_num_epochs", 9999) > MAX_TRAIN_EPOCHS:
                                kw["max_num_epochs"] = MAX_TRAIN_EPOCHS
                            kw.setdefault("training_batch_size", TRAIN_BATCH_SIZE)
                            print(f"    [Epoch cap: {MAX_TRAIN_EPOCHS}]")
                            return orig(*a, **kw)

                        return _inner

                    inf.train = _cap()
                    _patched_train_ok = True
                    print(f"    ✓ Patched via self.{attr}.train")
                    break

        _sp.ASNPE.__init__ = _patched_init
        if not _patched_train_ok:
            print("    (will patch at init time)")
            _patched_train_ok = True
    except Exception as e:
        print(f"    ✗ seqinf fallback: {e}")

if not _patched_train_ok:
    print("  ⚠ Epoch cap NOT applied — training may be slow on later rounds")

from seqinf.methods.posterior import ASNPE
from seqinf import BayesianInferenceDiagnostic


# ============================================================
# HELPERS
# ============================================================
def run_with_timeout(func, timeout_sec=180, label="operation"):
    result = [None]
    error = [None]
    completed = threading.Event()

    def target():
        try:
            result[0] = func()
        except Exception as e:
            error[0] = e
        finally:
            completed.set()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    completed.wait(timeout=timeout_sec)
    if not completed.is_set():
        raise TimeoutError(f"{label} timed out after {timeout_sec}s")
    if error[0] is not None:
        raise error[0]
    return result[0]


class SimProgress:
    def __init__(self, total_budget, report_every=25):
        self.count = 0
        self.fail = 0
        self.start_time = None
        self.total_budget = total_budget
        self.report_every = report_every
        self._lock = threading.Lock()
        self._last_print = 0
        # Round detection: track sim count at each "batch boundary"
        self._round_boundaries = []
        self._last_boundary_count = 0

    def tick(self, success=True):
        with self._lock:
            now = time.time()
            if self.start_time is None:
                self.start_time = now
            if success:
                self.count += 1
            else:
                self.fail += 1

            if (
                success
                and self.count % self.report_every == 0
                and (now - self._last_print) > 2.0
            ):
                self._last_print = now
                self._show(now)

    def _show(self, now):
        elapsed = now - self.start_time
        rate = self.count / (elapsed / 60) if elapsed > 0 else 0
        eta = (self.total_budget - self.count) / (rate / 60) if rate > 0 else 0
        pct = self.count / self.total_budget * 100

        bar_len = 30
        filled = int(bar_len * self.count / max(self.total_budget, 1))
        bar = "█" * filled + "░" * (bar_len - filled)
        fail_s = f" │ fail={self.fail}" if self.fail else ""

        # Detect round from training log
        current_round = len(_training_log["rounds"])

        print(
            f"    [{bar}] {self.count}/{self.total_budget} ({pct:.0f}%) │ "
            f"R{current_round} │ {rate:.0f}/min │ "
            f"ETA: {eta / 60:.1f}m ({eta / 3600:.1f}h){fail_s}",
            flush=True,
        )

    def summary(self):
        with self._lock:
            el = time.time() - self.start_time if self.start_time else 0
            return {
                "sims": self.count,
                "fail": self.fail,
                "time_s": el,
                "rate": self.count / (el / 60) if el > 0 else 0,
            }


# ============================================================
# CONFIG — 1000 SIMS (FAIR COMPARISON)
# ============================================================
SUMO_BINARY = "sumo"
BASE_DIR = Path(".")
SIM_TIMEOUT = 120

N_ROUNDS = 4
N_SAMPLES_PER_ROUND = 250
N_WORKERS = 15
N_POSTERIOR_SAMPLES = 50_000

PARAM_NAMES = ["speedFactor", "speedDev", "sigma", "tau"]
PRIOR_LOW = torch.tensor([0.5, 0.0, 0.0, 0.5], dtype=torch.float32)
PRIOR_HIGH = torch.tensor([1.3, 0.3, 1.0, 3.0], dtype=torch.float32)

progress = SimProgress(N_ROUNDS * N_SAMPLES_PER_ROUND, report_every=25)

# ============================================================
# LOAD
# ============================================================
print("\n" + "=" * 65)
print("  ASNPE v6 — FULL 1,000 SIM RUN (fair SNPE comparison)")
print(
    f"  Budget: {N_ROUNDS} × {N_SAMPLES_PER_ROUND} = {N_ROUNDS * N_SAMPLES_PER_ROUND}"
)
print("=" * 65)

print("\nLoading data...")
EDGE_MAX_SPEEDS, EDGE_LENGTHS = get_edge_max_speeds("osm.net.xml")
print(f"  ✓ {len(EDGE_MAX_SPEEDS)} edges")

feat_sel = torch.load("snpe_feature_selection.pt", weights_only=False)
KEPT_INDICES = feat_sel["kept_original_indices"]
FEAT_NAMES = feat_sel["final_names"]
N_FEATURES = len(KEPT_INDICES)
print(f"  ✓ {N_FEATURES} features: {FEAT_NAMES}")

x_obs_full = torch.load("x_obs.pt", weights_only=True).float()
x_obs = x_obs_full[KEPT_INDICES]
print(f"  ✓ x_obs: {x_obs.shape}")


# ============================================================
# SIMULATOR
# ============================================================
def sumo_simulator(theta):
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
        progress.tick(success=False)
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
                    "-c",
                    str((BASE_DIR / "sbi_peak.sumocfg").resolve()),
                    "--additional-files",
                    f"{vtype_path.resolve()},{edgedata_add.resolve()}",
                    "--seed",
                    str(np.random.randint(100000)),
                ],
                capture_output=True,
                timeout=SIM_TIMEOUT,
                cwd=str(BASE_DIR.resolve()),
            )

            if result.returncode == 0 and edgedata_output.exists():
                edge_data = parse_edge_data(str(edgedata_output))
                x_full = summary_statistics_sumo(
                    edge_data, EDGE_MAX_SPEEDS, EDGE_LENGTHS
                )
                _cleanup(sim_dir)

                if (
                    x_full is not None
                    and len(x_full) == 57
                    and np.all(np.isfinite(x_full))
                ):
                    progress.tick(success=True)
                    return torch.tensor(x_full[KEPT_INDICES], dtype=torch.float32)

            if attempt == 0:
                time.sleep(0.3)
        except subprocess.TimeoutExpired:
            if attempt == 0:
                time.sleep(0.3)
        except Exception:
            pass

    _cleanup(sim_dir)
    progress.tick(success=False)
    return torch.zeros(N_FEATURES, dtype=torch.float32)


def _cleanup(d):
    try:
        for f in d.glob("*"):
            f.unlink(missing_ok=True)
        d.rmdir()
    except OSError:
        pass


# ============================================================
# ROBUST SAMPLING
# ============================================================
def sample_posterior_robust(posterior, prior, x_obs, n, plo, phi):
    print("  [1/3] Direct (90s)...")
    try:
        s = run_with_timeout(lambda: posterior.sample((n,)), 90, "Direct")
        print("  ✓ Direct OK")
        return s.numpy()
    except (TimeoutError, Exception) as e:
        print(f"  ✗ {e}")

    print("  [2/3] Importance resampling (500k)...")
    try:
        cands = prior.sample((500_000,))
        with torch.no_grad():
            for sig in [
                lambda t: posterior.log_prob(t, x=x_obs),
                lambda t: posterior.log_prob(t, x=x_obs.unsqueeze(0)),
                lambda t: posterior.log_prob(t),
            ]:
                try:
                    lp = sig(cands)
                    break
                except TypeError:
                    continue
        valid = torch.isfinite(lp)
        if valid.sum() < 1000:
            raise ValueError(f"Only {valid.sum()} valid")
        lp = torch.where(valid, lp, torch.tensor(float("-inf")))
        w = torch.exp(lp - lp[valid].max())
        w /= w.sum()
        idx = torch.multinomial(w, n, replacement=True)
        s = cands[idx] + (phi - plo) * 0.005 * torch.randn(n, 4)
        s = torch.clamp(s, plo, phi)
        ess = 1.0 / (w[w > 0] ** 2).sum().item()
        print(f"  ✓ Importance OK (ESS={ess:.0f})")
        return s.numpy()
    except Exception as e:
        print(f"  ✗ {e}")

    print("  [3/3] KDE...")
    try:
        sm = run_with_timeout(lambda: posterior.sample((2000,)), 180, "Small")
        kde = gaussian_kde(sm.numpy().T)
        exp = kde.resample(n).T
        for i in range(4):
            exp[:, i] = np.clip(exp[:, i], plo[i].item(), phi[i].item())
        print("  ✓ KDE OK")
        return exp
    except Exception as e:
        print(f"  ✗ {e}")

    print("  ⚠ Returning prior")
    return prior.sample((n,)).numpy()


# ============================================================
# MAIN
# ============================================================
def main():
    assert (BASE_DIR / "osm.net.xml").exists()
    assert (BASE_DIR / "sbi_peak.sumocfg").exists()
    assert (BASE_DIR / "routes_peak_novtype.rou.xml").exists()
    (BASE_DIR / "asnpe_runs").mkdir(exist_ok=True)

    prior = BoxUniform(low=PRIOR_LOW, high=PRIOR_HIGH)
    total = N_ROUNDS * N_SAMPLES_PER_ROUND

    # Test
    print("\n┌─ Simulator Test ──────────────────────")
    t0 = time.time()
    tx = sumo_simulator(torch.tensor([0.9, 0.1, 0.5, 1.5]))
    dt = time.time() - t0
    print(
        f"│ shape={tx.shape}, time={dt:.1f}s, non-zero={tx.count_nonzero().item()}/{N_FEATURES}"
    )
    assert tx.shape == (N_FEATURES,)
    print(f"└───────────────────────────────────────")

    progress.count = 0
    progress.fail = 0
    progress.start_time = None

    est_sim = total / 34  # ~34 sim/min from your runs
    est_train = N_ROUNDS * 2  # ~2 min per round training with cap
    print(f"\n┌─ Configuration ──────────────────────────────")
    print(f"│ Method:           ASNPE (Active Sequential NPE)")
    print(f"│ Density est:      MAF")
    print(f"│ Rounds:           {N_ROUNDS}")
    print(f"│ Sims/round:       {N_SAMPLES_PER_ROUND}")
    print(f"│ Total budget:     {total} (same as SNPE)")
    print(f"│ Workers:          {N_WORKERS}")
    print(f"│ Max epochs/round: {MAX_TRAIN_EPOCHS}")
    print(f"│ Batch size:       {TRAIN_BATCH_SIZE}")
    print(f"│")
    print(f"│ Est. sim time:    ~{est_sim:.0f} min")
    print(f"│ Est. train time:  ~{est_train:.0f} min")
    print(
        f"│ Est. total:       ~{(est_sim + est_train):.0f} min "
        f"(~{(est_sim + est_train) / 60:.1f} hours)"
    )
    print(f"│")
    print(f"│ KEY: Single asnpe.run(num_rounds={N_ROUNDS})")
    print(f"│      Proper sequential posterior updates")
    print(f"└──────────────────────────────────────────────")

    # Create
    print("\nCreating ASNPE...")
    asnpe = ASNPE(
        simulator=sumo_simulator,
        prior=prior,
        density_estimator="maf",
        num_workers=N_WORKERS,
    )
    print("  ✓ Created")

    # ============================================================
    # RUN — SINGLE CALL
    # ============================================================
    print(f"\n{'=' * 65}")
    print(
        f"  RUNNING: asnpe.run(num_rounds={N_ROUNDS}, "
        f"num_samples={N_SAMPLES_PER_ROUND})"
    )
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Fire and forget — check back in ~{(est_sim + est_train) / 60:.1f} hours")
    print(f"{'=' * 65}\n")

    t_start = time.time()

    with joblib.parallel_backend("threading", n_jobs=N_WORKERS):
        asnpe.run(
            num_rounds=N_ROUNDS,
            num_samples=N_SAMPLES_PER_ROUND,
            x_o=x_obs,
            seed=42,
        )

    total_time = time.time() - t_start
    stats = progress.summary()

    print(f"\n{'=' * 65}")
    print(f"  ALL ROUNDS COMPLETE")
    print(
        f"  Sims: {stats['sims']} │ Failed: {stats['fail']} │ "
        f"Time: {total_time / 60:.1f}m ({total_time / 3600:.1f}h) │ "
        f"Rate: {stats['rate']:.0f}/min"
    )
    print(f"{'=' * 65}")

    # ---- Immediate save ----
    try:
        with open("asnpe_inference.pkl", "wb") as f:
            pickle.dump(asnpe, f)
        print("  ✓ asnpe_inference.pkl")
    except Exception as e:
        print(f"  ✗ pickle: {e}")

    # ---- Posterior ----
    print(f"\n{'=' * 65}")
    print(f"  POSTERIOR SAMPLING ({N_POSTERIOR_SAMPLES:,})")
    print(f"{'=' * 65}")

    posterior = asnpe.proposal
    print(f"  Type: {type(posterior).__name__}")

    samples_np = sample_posterior_robust(
        posterior, prior, x_obs, N_POSTERIOR_SAMPLES, PRIOR_LOW, PRIOR_HIGH
    )
    for i in range(4):
        samples_np[:, i] = np.clip(
            samples_np[:, i], PRIOR_LOW[i].item(), PRIOR_HIGH[i].item()
        )

    # ---- Results ----
    print(
        f"\n  {'Param':>15} │ {'Mean':>8} │ {'Median':>8} │ {'Std':>8} │ "
        f"{'5%':>8} │ {'95%':>8} │ {'MAP':>8}"
    )
    print(f"  {'─' * 75}")

    map_est = []
    for i, nm in enumerate(PARAM_NAMES):
        c = samples_np[:, i]
        q05, q95 = np.percentile(c, [5, 95])
        try:
            kde = gaussian_kde(c)
            g = np.linspace(PRIOR_LOW[i].item(), PRIOR_HIGH[i].item(), 1000)
            mp = g[kde(g).argmax()]
        except:
            mp = np.median(c)
        map_est.append(mp)
        print(
            f"  {nm:>15} │ {c.mean():8.4f} │ {np.median(c):8.4f} │ "
            f"{c.std():8.4f} │ {q05:8.4f} │ {q95:8.4f} │ {mp:8.4f}"
        )

    # ---- Per-round data ----
    print("\n--- Per-Round Data ---")
    all_thetas, all_xs = [], []
    for r in range(N_ROUNDS):
        for args in [(r,), (r, 42)]:
            try:
                rt = asnpe.get_round_thetas(*args)
                rx = asnpe.get_round_xs(*args)
                all_thetas.append(rt)
                all_xs.append(rx)
                rn = rt.numpy() if isinstance(rt, torch.Tensor) else np.array(rt)
                means = ", ".join(f"{rn[:, i].mean():.3f}" for i in range(4))
                stds = ", ".join(f"{rn[:, i].std():.3f}" for i in range(4))
                print(f"  R{r}: n={rn.shape[0]:4d}  mean=[{means}]  std=[{stds}]")
                break
            except Exception as e:
                if args[-1] == 42:
                    print(f"  R{r}: {e}")

    # ---- Training summary ----
    if _training_log["rounds"]:
        print("\n--- Training Summary ---")
        print(
            f"  {'Round':>5} │ {'Epochs':>8} │ {'Train Loss':>10} │ "
            f"{'Val Loss':>10} │ {'Time':>8} │ {'Data':>6}"
        )
        print(f"  {'─' * 60}")
        for tm in _training_log["rounds"]:
            ri = tm.get("round", "?")
            ep = tm.get("epochs", "?")
            ep_str = f"{ep}" if ep != "?" else "?"
            if ep == MAX_TRAIN_EPOCHS:
                ep_str += " ⚠CAP"
            tl = (
                f"{tm['final_train_loss']:.4f}"
                if tm.get("final_train_loss") is not None
                else "?"
            )
            vl = (
                f"{tm['best_val_loss']:.4f}"
                if tm.get("best_val_loss") is not None
                else "?"
            )
            tt = f"{tm.get('train_time_s', 0):.1f}s"
            nd = str(tm.get("n_data", "?"))
            print(f"  R{ri:>4} │ {ep_str:>8} │ {tl:>10} │ {vl:>10} │ {tt:>8} │ {nd:>6}")

        total_train = sum(tm.get("train_time_s", 0) for tm in _training_log["rounds"])
        total_epochs = sum(tm.get("epochs", 0) or 0 for tm in _training_log["rounds"])
        print(f"\n  Total training: {total_train / 60:.1f} min, {total_epochs} epochs")

    # ---- PLOTS ----
    print("\n--- Plots ---")

    # 1. Pairplot
    try:
        fig, _ = pairplot(
            samples_np,
            labels=PARAM_NAMES,
            limits=list(zip(PRIOR_LOW.numpy(), PRIOR_HIGH.numpy())),
            figsize=(10, 10),
        )
        fig.suptitle(
            f"ASNPE Posterior ({total} sims, {N_ROUNDS} rounds)", fontsize=14, y=1.02
        )
        plt.savefig("asnpe_pairplot.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✓ asnpe_pairplot.png")
    except Exception as e:
        print(f"  ✗ pairplot: {e}")
        plt.close("all")

    # 2. Marginals
    try:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        for i, (nm, ax) in enumerate(zip(PARAM_NAMES, axes)):
            c = samples_np[:, i]
            lo, hi = PRIOR_LOW[i].item(), PRIOR_HIGH[i].item()
            ax.hist(
                c,
                bins=80,
                density=True,
                alpha=0.7,
                color="darkorange",
                edgecolor="white",
                linewidth=0.3,
            )
            ax.axhline(1 / (hi - lo), color="gray", ls="--", alpha=0.6, label="Prior")
            ax.axvline(c.mean(), color="red", lw=1.5, label=f"Mean={c.mean():.3f}")
            q05, q95 = np.percentile(c, [5, 95])
            ax.axvspan(q05, q95, alpha=0.15, color="red")
            ax.set_xlim(lo, hi)
            ax.set_title(nm)
            ax.legend(fontsize=8)
        plt.suptitle(f"ASNPE Marginals ({total} sims)", fontsize=13)
        plt.tight_layout()
        plt.savefig("asnpe_marginals.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✓ asnpe_marginals.png")
    except Exception as e:
        print(f"  ✗ marginals: {e}")
        plt.close("all")

    # 3. Round evolution
    if len(all_thetas) > 1:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            for i, (nm, ax) in enumerate(zip(PARAM_NAMES, axes)):
                lo, hi = PRIOR_LOW[i].item(), PRIOR_HIGH[i].item()
                for ri, rt in enumerate(all_thetas):
                    rn = rt.numpy() if isinstance(rt, torch.Tensor) else np.array(rt)
                    ax.hist(
                        rn[:, i],
                        bins=30,
                        density=True,
                        alpha=0.35,
                        color=plt.cm.viridis(ri / max(len(all_thetas) - 1, 1)),
                        label=f"R{ri}" if ri % 2 == 0 else None,
                    )
                ax.hist(
                    samples_np[:, i],
                    bins=50,
                    density=True,
                    alpha=0.3,
                    color="red",
                    label="Posterior",
                )
                ax.set_xlim(lo, hi)
                ax.set_title(nm)
                ax.legend(fontsize=6)
            plt.suptitle("ASNPE Round Evolution", fontsize=14)
            plt.tight_layout()
            plt.savefig("asnpe_round_evolution.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("  ✓ asnpe_round_evolution.png")
        except Exception as e:
            print(f"  ✗ evolution: {e}")
            plt.close("all")

    # 4. Convergence vs SNPE
    if len(all_thetas) > 1:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            axes = axes.flatten()
            try:
                snpe_r = dict(np.load("snpe_results.npz", allow_pickle=True))
                has_snpe = True
            except:
                has_snpe = False
            for i, (nm, ax) in enumerate(zip(PARAM_NAMES, axes)):
                ms = [
                    (rt.numpy() if isinstance(rt, torch.Tensor) else np.array(rt))[
                        :, i
                    ].mean()
                    for rt in all_thetas
                ]
                ss = [
                    (rt.numpy() if isinstance(rt, torch.Tensor) else np.array(rt))[
                        :, i
                    ].std()
                    for rt in all_thetas
                ]
                ax.plot(range(len(ms)), ms, "o-", color="darkorange", lw=2)
                ax.fill_between(
                    range(len(ms)),
                    [m - s for m, s in zip(ms, ss)],
                    [m + s for m, s in zip(ms, ss)],
                    alpha=0.2,
                    color="darkorange",
                )
                if has_snpe:
                    ax.axhline(
                        float(snpe_r["mean"][i]),
                        color="steelblue",
                        ls="--",
                        lw=1.5,
                        label=f"SNPE={float(snpe_r['mean'][i]):.3f}",
                    )
                ax.set_xlabel("Round")
                ax.set_ylabel(nm)
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
            plt.suptitle("ASNPE vs SNPE (both 1000 sims)", fontsize=14)
            plt.tight_layout()
            plt.savefig("asnpe_convergence.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("  ✓ asnpe_convergence.png")
        except Exception as e:
            print(f"  ✗ convergence: {e}")
            plt.close("all")

    # 5. Training curves
    rounds_with_losses = [
        tm for tm in _training_log["rounds"] if tm.get("train_losses")
    ]
    if rounds_with_losses:
        try:
            n_plots = min(len(rounds_with_losses), 10)
            fig, axes = plt.subplots(2, 5, figsize=(25, 8))
            axes = axes.flatten()
            for idx, tm in enumerate(rounds_with_losses[:n_plots]):
                ax = axes[idx]
                if tm.get("train_losses"):
                    ax.plot(
                        tm["train_losses"],
                        color="darkorange",
                        alpha=0.8,
                        lw=1,
                        label="Train",
                    )
                if tm.get("val_losses"):
                    ax.plot(
                        tm["val_losses"],
                        color="steelblue",
                        alpha=0.8,
                        lw=1,
                        label="Val",
                    )
                ax.set_title(
                    f"R{tm.get('round', idx)} ({tm.get('epochs', '?')}ep)", fontsize=9
                )
                ax.legend(fontsize=6)
                ax.grid(alpha=0.3)
                ax.set_xlabel("Epoch", fontsize=8)
            for idx in range(n_plots, 10):
                axes[idx].set_visible(False)
            plt.suptitle("Training Curves per Round", fontsize=14)
            plt.tight_layout()
            plt.savefig("asnpe_training_curves.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("  ✓ asnpe_training_curves.png")
        except Exception as e:
            print(f"  ✗ training curves: {e}")
            plt.close("all")

    # 6. Budget
    if len(all_thetas) > 0:
        try:
            sizes = [
                (t.numpy() if isinstance(t, torch.Tensor) else np.array(t)).shape[0]
                for t in all_thetas
            ]
            cum = np.cumsum(sizes)
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
            a1.bar(range(len(sizes)), sizes, color="darkorange", alpha=0.7)
            a1.set_xlabel("Round")
            a1.set_ylabel("Sims")
            a1.set_title("Per round")
            a2.plot(range(len(cum)), cum, "o-", color="darkorange", lw=2, ms=6)
            a2.axhline(total, color="gray", ls="--", alpha=0.5, label=f"Budget={total}")
            a2.set_xlabel("Round")
            a2.set_ylabel("Cumulative")
            a2.set_title("Total")
            a2.legend()
            plt.tight_layout()
            plt.savefig("asnpe_budget.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("  ✓ asnpe_budget.png")
        except Exception as e:
            print(f"  ✗ budget: {e}")
            plt.close("all")

    # ---- Diagnostics ----
    print("\n--- seqinf diagnostics ---")
    try:
        diag = BayesianInferenceDiagnostic(asnpe, num_workers=1)
        for fn, func in [
            ("asnpe_diag_thetas.png", lambda: diag.plot_run_thetas()),
            ("asnpe_diag_proposals.png", lambda: diag.plot_run_proposals()),
        ]:
            try:
                run_with_timeout(func, 120, fn)
                plt.savefig(fn, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  ✓ {fn}")
            except (TimeoutError, Exception) as e:
                print(f"  ✗ {fn}: {e}")
                plt.close("all")
    except Exception as e:
        print(f"  ✗ {e}")

    # ---- SAVE ----
    print(f"\n{'=' * 65}")
    print(f"  SAVING")
    print(f"{'=' * 65}")

    try:
        with open("asnpe_posterior.pkl", "wb") as f:
            pickle.dump(posterior, f)
        print("  ✓ asnpe_posterior.pkl")
    except Exception as e:
        print(f"  ✗ {e}")

    torch.save(torch.tensor(samples_np, dtype=torch.float32), "asnpe_samples.pt")
    print("  ✓ asnpe_samples.pt")

    results = {
        "param_names": PARAM_NAMES,
        "map": np.array(map_est),
        "mean": samples_np.mean(axis=0),
        "median": np.median(samples_np, axis=0),
        "std": samples_np.std(axis=0),
        "q05": np.percentile(samples_np, 5, axis=0),
        "q95": np.percentile(samples_np, 95, axis=0),
        "n_rounds": N_ROUNDS,
        "n_per_round": N_SAMPLES_PER_ROUND,
        "total_sims": progress.count,
        "total_failures": progress.fail,
        "total_time_s": total_time,
        "n_features": N_FEATURES,
    }
    np.savez("asnpe_results.npz", **results)
    print("  ✓ asnpe_results.npz")

    if all_thetas:
        torch.save(
            {
                "thetas": [
                    t if isinstance(t, torch.Tensor) else torch.tensor(t)
                    for t in all_thetas
                ],
                "xs": [
                    x if isinstance(x, torch.Tensor) else torch.tensor(x)
                    for x in all_xs
                ],
            },
            "asnpe_round_data.pt",
        )
        print("  ✓ asnpe_round_data.pt")

    if _training_log["rounds"]:
        with open("asnpe_training_log.pkl", "wb") as f:
            pickle.dump(_training_log, f)
        print("  ✓ asnpe_training_log.pkl")

    # ---- FINAL ----
    print(f"\n{'=' * 65}")
    print(f"  ASNPE FINAL RESULTS (1000 sims)")
    print(f"{'=' * 65}")
    print(
        f"  Sims: {progress.count} │ Failed: {progress.fail} │ "
        f"Time: {total_time / 60:.1f}m ({total_time / 3600:.1f}h)"
    )

    print(f"\n  {'Param':>15} │ {'MAP':>8} │ {'Mean±Std':>15} │ {'90% CI':>18}")
    print(f"  {'─' * 60}")
    for i, nm in enumerate(PARAM_NAMES):
        c = samples_np[:, i]
        q05, q95 = np.percentile(c, [5, 95])
        print(
            f"  {nm:>15} │ {map_est[i]:8.4f} │ "
            f"{c.mean():7.4f} ± {c.std():.4f} │ [{q05:.4f}, {q95:.4f}]"
        )

    try:
        snpe_r = dict(np.load("snpe_results.npz", allow_pickle=True))
        print(f"\n  ┌─ ASNPE vs SNPE (both 1000 sims) ─────────────────")
        print(f"  │ {'Param':>15} │ {'SNPE Std':>9} │ {'ASNPE Std':>9} │ {'Winner':>8}")
        print(f"  │ {'─' * 47}")
        wins = 0
        for i, nm in enumerate(PARAM_NAMES):
            ss = float(snpe_r["std"][i])
            sa = samples_np[:, i].std()
            w = "ASNPE" if sa < ss else "SNPE"
            if w == "ASNPE":
                wins += 1
            print(f"  │ {nm:>15} │ {ss:9.4f} │ {sa:9.4f} │ {w:>8}")
        print(f"  │")
        print(f"  │ Score: ASNPE wins {wins}/4 parameters (equal budget)")
        print(f"  └────────────────────────────────────────────────")
    except:
        pass

    print(f"\n  ✓ Next: python compare_methods.py")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
