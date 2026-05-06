"""
Partie 4 — Ensemble Stacking : TabM† + XGBoost + LightGBM
==========================================================
Architecture :
  Niveau 0 (base learners) :
    - TabM† (meilleure variante ablation, hyperparams optimisés par sweep)
    - XGBoost (gradient boosting)
    - LightGBM (gradient boosting rapide)

  Niveau 1 (méta-modèle) :
    - Moyenne pondérée optimisée + offsets par classe (QWK)

Stratégie : K-Fold cross-validation pour générer les OOF predictions,
            exécution en sous-processus séparés pour éviter conflits PyTorch/XGB.
"""

from __future__ import annotations
import json
import sys
import time
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE        = Path(__file__).parent
RESULTS_DIR = BASE / "results"
STACK_DIR   = RESULTS_DIR / "stacking"
STACK_DIR.mkdir(parents=True, exist_ok=True)

SEED    = 42
N_FOLDS = 5

# ── Worker XGBoost (lancé en sous-processus) ──────────────────────────────────
XGB_WORKER = '''
import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

data = json.loads(sys.stdin.read())
X = np.array(data["X"]); y = np.array(data["y"])
seed = data["seed"]; n_folds = data["n_folds"]

params = dict(n_estimators=800, max_depth=6, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
              reg_alpha=0.1, reg_lambda=1.0, objective="reg:squarederror",
              random_state=seed, n_jobs=-1, verbosity=0)

skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
oof = np.zeros(len(y))
for fold, (tr, val) in enumerate(skf.split(X, y.astype(int))):
    m = xgb.XGBRegressor(**params)
    m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], verbose=False)
    oof[val] = m.predict(X[val])
    q = float(cohen_kappa_score(y[val].astype(int),
              np.clip(np.round(oof[val]),1,8).astype(int), weights="quadratic"))
    print(f"   [XGBoost] Fold {fold+1}/{n_folds} QWK={q:.4f}", flush=True)

print(json.dumps(oof.tolist()))
'''

LGB_WORKER = '''
import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

data = json.loads(sys.stdin.read())
X = np.array(data["X"]); y = np.array(data["y"])
seed = data["seed"]; n_folds = data["n_folds"]

params = dict(n_estimators=800, max_depth=6, learning_rate=0.05,
              num_leaves=63, subsample=0.8, colsample_bytree=0.8,
              min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
              objective="regression", random_state=seed, n_jobs=-1, verbose=-1)

skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
oof = np.zeros(len(y))
for fold, (tr, val) in enumerate(skf.split(X, y.astype(int))):
    m = lgb.LGBMRegressor(**params)
    m.fit(X[tr], y[tr],
          eval_set=[(X[val], y[val])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof[val] = m.predict(X[val])
    q = float(cohen_kappa_score(y[val].astype(int),
              np.clip(np.round(oof[val]),1,8).astype(int), weights="quadratic"))
    print(f"   [LightGBM] Fold {fold+1}/{n_folds} QWK={q:.4f}", flush=True)

print(json.dumps(oof.tolist()))
'''

TABM_WORKER = '''
import sys, json, numpy as np, warnings, time
warnings.filterwarnings("ignore")
import os; os.environ["PYTHONPATH"] = sys.argv[1]
sys.path.insert(0, sys.argv[1])
from config_ablation import TRAIN_PATH, AblationVariantConfig
from model_ablation import build_model
from metrics_ablation import qwk, optimize_offsets, apply_offsets
from sklearn.model_selection import StratifiedKFold
import pandas as pd

data = json.loads(sys.stdin.read())
X = np.array(data["X"]); y = np.array(data["y"])
seed = data["seed"]; n_folds = data["n_folds"]

cfg = AblationVariantConfig(
    variant_id="tabm_embed", arch_type="tabm", k=32,
    n_blocks=3, d_block=512, dropout=0.229, lr=0.001120,
    weight_decay=0.000337, batch_size=1024, epochs=50,
    early_stopping_patience=10, use_embeddings=True,
    n_bins=48, d_embedding=32, label="TabM† sweep optimal",
)

skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
oof = np.zeros(len(y))
for fold, (tr, val) in enumerate(skf.split(X, y.astype(int))):
    t0 = time.time()
    m = build_model(X[tr].shape[1], cfg)
    m.fit(X[tr], y[tr], X[val], y[val])
    oof[val] = m.predict(X[val])
    q = qwk(oof[val], y[val])
    print(f"   [TabM†] Fold {fold+1}/{n_folds} QWK={q:.4f} ({time.time()-t0:.0f}s)", flush=True)

print(json.dumps(oof.tolist()))
'''


def run_worker(script: str, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Lance un worker dans un sous-processus Python indépendant."""
    payload = json.dumps({
        "X": X.tolist(), "y": y.tolist(),
        "seed": SEED, "n_folds": N_FOLDS,
    })
    result = subprocess.run(
        [sys.executable, "-c", script, str(BASE)],
        input=payload.encode(),
        capture_output=False,
        stdout=subprocess.PIPE,
        stderr=sys.stdout.fileno(),
        timeout=7200,
    )
    # La dernière ligne stdout contient le JSON des OOF
    lines = result.stdout.decode().strip().splitlines()
    for line in reversed(lines):
        try:
            return np.array(json.loads(line))
        except Exception:
            continue
    raise RuntimeError(f"Worker n'a pas retourné de prédictions.\nstdout={result.stdout.decode()[-500:]}")


def load_data():
    from config_ablation import TRAIN_PATH
    print("📥 Chargement des données...")
    df = pd.read_csv(TRAIN_PATH)
    X = df.drop(columns=["Response"]).select_dtypes(include=[np.number]).fillna(0)
    y = df["Response"].values.astype(float)
    print(f"   Shape: {X.shape}")
    return X.values, y


def qwk(preds, labels):
    from sklearn.metrics import cohen_kappa_score
    labels = np.array(labels).astype(int)
    preds  = np.clip(np.round(np.array(preds)), 1, 8).astype(int)
    return float(cohen_kappa_score(labels, preds, weights="quadratic"))


def optimize_offsets(preds, labels):
    from scipy.optimize import minimize_scalar
    preds  = np.array(preds, dtype=float)
    labels = np.array(labels, dtype=float)
    adj    = preds.copy()
    offsets = np.zeros(8)
    for j in [6, 4, 5, 3, 2, 1, 7, 0]:
        def neg_qwk(delta):
            tmp = adj.copy(); tmp[adj >= j + 0.5] += delta
            return -qwk(tmp, labels)
        res = minimize_scalar(neg_qwk, bounds=(-1, 1), method="bounded")
        offsets[j] = res.x
        adj[preds >= j + 0.5] += res.x
    return offsets

def apply_offsets(preds, offsets):
    adj = np.array(preds, dtype=float)
    for j in range(8):
        adj[preds >= j + 0.5] += offsets[j]
    return adj


def evaluate(preds, y, name):
    off  = optimize_offsets(preds, y)
    adj  = apply_offsets(preds, off)
    return {
        "qwk_raw":    round(qwk(preds, y), 4),
        "qwk_offset": round(qwk(adj, y),   4),
    }


def plot_results(results, oof_tabm, oof_xgb, oof_lgb, meta_preds, y):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Ensemble Stacking — TabM† + XGBoost + LightGBM", fontsize=14, fontweight="bold")

    models = list(results.keys())
    raw    = [results[m]["qwk_raw"]    for m in models]
    offset = [results[m]["qwk_offset"] for m in models]
    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
    x = np.arange(len(models))
    ax = axes[0]
    ax.bar(x - 0.2, raw,    0.38, label="QWK brut",       color=colors, alpha=0.5)
    ax.bar(x + 0.2, offset, 0.38, label="QWK + offsets",  color=colors, alpha=0.9)
    for i, o in enumerate(offset):
        ax.text(i + 0.2, o + 0.001, f"{o:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(0.6608, color="navy", linestyle="--", linewidth=1.5, alpha=0.7, label="Best sweep (0.6608)")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=10)
    ax.set_ylim(0.59, 0.72); ax.set_ylabel("QWK")
    ax.set_title("Comparaison des modèles"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.scatter(oof_tabm[:2000], oof_xgb[:2000], alpha=0.2, s=10, color="#9b59b6")
    ax.set_xlabel("TabM† prédictions"); ax.set_ylabel("XGBoost prédictions")
    ax.set_title("Corrélation TabM† vs XGBoost")
    r = np.corrcoef(oof_tabm, oof_xgb)[0, 1]
    ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes, fontsize=11, va="top",
            color="#9b59b6", fontweight="bold")

    ax = axes[2]
    ax.hist(oof_tabm,   bins=40, alpha=0.5, label="TabM†",   color="#e74c3c", density=True)
    ax.hist(meta_preds, bins=40, alpha=0.5, label="Stacking", color="#2ecc71", density=True)
    ax.set_xlabel("Prédiction"); ax.set_ylabel("Densité")
    ax.set_title("Distribution : TabM† vs Stacking"); ax.legend()

    fig.tight_layout()
    path = STACK_DIR / "stacking_results.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def main():
    t0 = time.time()
    print("=" * 60)
    print("  ENSEMBLE STACKING — TabM† + XGBoost + LightGBM")
    print("=" * 60)

    X, y = load_data()

    # ── Niveau 0 : Base Learners (sous-processus séparés) ──
    print("\n🔵 Entraînement TabM† (K-Fold OOF)...")
    oof_tabm = run_worker(TABM_WORKER, X, y)
    r_tabm = evaluate(oof_tabm, y, "TabM†")
    print(f"   → QWK TabM†    brut={r_tabm['qwk_raw']:.4f}  +offsets={r_tabm['qwk_offset']:.4f}")

    print("\n🟠 Entraînement XGBoost (K-Fold OOF)...")
    oof_xgb = run_worker(XGB_WORKER, X, y)
    r_xgb = evaluate(oof_xgb, y, "XGBoost")
    print(f"   → QWK XGBoost  brut={r_xgb['qwk_raw']:.4f}  +offsets={r_xgb['qwk_offset']:.4f}")

    print("\n🟢 Entraînement LightGBM (K-Fold OOF)...")
    oof_lgb = run_worker(LGB_WORKER, X, y)
    r_lgb = evaluate(oof_lgb, y, "LightGBM")
    print(f"   → QWK LightGBM brut={r_lgb['qwk_raw']:.4f}  +offsets={r_lgb['qwk_offset']:.4f}")

    # Corrélations
    print(f"\n📊 Corrélations OOF :")
    print(f"   TabM† vs XGBoost   : r = {np.corrcoef(oof_tabm, oof_xgb)[0,1]:.3f}")
    print(f"   TabM† vs LightGBM  : r = {np.corrcoef(oof_tabm, oof_lgb)[0,1]:.3f}")
    print(f"   XGBoost vs LightGBM: r = {np.corrcoef(oof_xgb,  oof_lgb)[0,1]:.3f}")

    # ── Niveau 1 : Combinaison par recherche de poids optimaux ──
    print("\n🔴 Optimisation des poids de l'ensemble...")
    from scipy.optimize import minimize

    def neg_qwk_ensemble(w):
        w = np.array(w); w = np.abs(w) / np.sum(np.abs(w))
        meta = w[0]*oof_tabm + w[1]*oof_xgb + w[2]*oof_lgb
        off  = optimize_offsets(meta, y)
        adj  = apply_offsets(meta, off)
        return -qwk(adj, y)

    best_score, best_w = 0, [1/3, 1/3, 1/3]
    for w0 in [0.5, 0.6, 0.4, 0.7]:
        for w1 in [0.3, 0.2, 0.25]:
            w2 = 1 - w0 - w1
            if w2 <= 0: continue
            s = -neg_qwk_ensemble([w0, w1, w2])
            if s > best_score:
                best_score, best_w = s, [w0, w1, w2]
    best_w = np.abs(best_w) / np.sum(np.abs(best_w))
    print(f"   Poids optimaux : TabM†={best_w[0]:.3f}, XGB={best_w[1]:.3f}, LGB={best_w[2]:.3f}")

    meta_preds = best_w[0]*oof_tabm + best_w[1]*oof_xgb + best_w[2]*oof_lgb
    r_stack = evaluate(meta_preds, y, "Stacking")

    # ── Résultats ──
    results = {"TabM†": r_tabm, "XGBoost": r_xgb, "LightGBM": r_lgb, "Stacking": r_stack}

    print("\n" + "=" * 65)
    print(f"  {'Modèle':<20} {'QWK brut':>10} {'QWK + offsets':>14}  {'vs TabM† sweep':>15}")
    print("-" * 65)
    ref = 0.6608
    for name, r in results.items():
        gain = r["qwk_offset"] - ref
        best_mark = " 🏆" if r["qwk_offset"] == max(v["qwk_offset"] for v in results.values()) else ""
        print(f"  {name:<20} {r['qwk_raw']:>10.4f} {r['qwk_offset']:>14.4f}  ({gain:>+.4f}){best_mark}")
    print("=" * 65)

    # ── Sauvegarde ──
    plot_results(results, oof_tabm, oof_xgb, oof_lgb, meta_preds, y)

    summary = {
        "results":      results,
        "weights":      {"tabm": float(best_w[0]), "xgb": float(best_w[1]), "lgb": float(best_w[2])},
        "correlations": {
            "tabm_xgb": float(np.corrcoef(oof_tabm, oof_xgb)[0,1]),
            "tabm_lgb": float(np.corrcoef(oof_tabm, oof_lgb)[0,1]),
            "xgb_lgb":  float(np.corrcoef(oof_xgb,  oof_lgb)[0,1]),
        },
        "elapsed_min": round((time.time()-t0)/60, 1),
    }
    (STACK_DIR / "stacking_summary.json").write_text(json.dumps(summary, indent=2))

    # Sauvegarde des OOF pour le rapport
    np.save(STACK_DIR / "oof_tabm.npy", oof_tabm)
    np.save(STACK_DIR / "oof_xgb.npy",  oof_xgb)
    np.save(STACK_DIR / "oof_lgb.npy",  oof_lgb)
    np.save(STACK_DIR / "oof_meta.npy", meta_preds)
    np.save(STACK_DIR / "labels.npy",   y)

    print(f"\n✅ Résultats sauvegardés dans {STACK_DIR}")
    print(f"   Durée totale : {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

