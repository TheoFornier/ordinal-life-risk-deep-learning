from __future__ import annotations
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

from models.tabm_model import TabMModel
from config import TabMConfig, DataConfig
from metrics import optimize_offsets, apply_offsets, qwk
from results_utils import save_confusion_matrix_plot, save_per_class_accuracy_plot
from train import load_data, load_test_data

DL_RUN = "results/train_clean_tabm_20260430_234510"
BOOST = "xgboost" # "catboost" or "xgboost"
OUT = "../submission_ensemble.csv"

MODELS_DIR = "../models_saved"


def main() -> None:
    dl_cfg = DataConfig()

    X, y = load_data(dl_cfg.train_path)
    test_ids, X_test = load_test_data(dl_cfg.test_path)

    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.astype(int)
    )

    # Boosting predictions
    print(f"Loading {BOOST} model...")
    if BOOST == "catboost":
        boost_model = CatBoostRegressor()
        boost_model.load_model(os.path.join(MODELS_DIR, "cat_model.cbm"))
        boost_val = boost_model.predict(X_val)
        boost_test = boost_model.predict(X_test)
    else:
        boost_model = xgb.Booster()
        boost_model.load_model(os.path.join(MODELS_DIR, "xgb_model.json"))
        boost_val = boost_model.predict(xgb.DMatrix(X_val))
        boost_test = boost_model.predict(xgb.DMatrix(X_test))

    print(f"Boosting val QWK (raw): {qwk(boost_val, y_val):.4f}")

    # Deep learning predictions
    print(f"Loading TabM model from {DL_RUN}...")
    dl_model = TabMModel(input_dim=X.shape[1], config=TabMConfig())
    dl_model.load(os.path.join(DL_RUN, "best_model.pt"))
    dl_val = dl_model.predict(X_val)
    dl_test = dl_model.predict(X_test)

    print(f"TabM val QWK (raw): {qwk(dl_val, y_val):.4f}")

    # Create timestamped results directory for this ensemble run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(dl_cfg.results_dir, f"ensemble_{BOOST}_vs_dl_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Ensemble results dir: {run_dir}")

    # Grid search blend weight alpha (boost=alpha, dl=1-alpha)
    print(f"\n{'alpha':>6}  {'QWK (with offsets)':>20}")
    print("-" * 30)
    best_alpha, best_score = 0.5, -1.0
    alpha_results = []
    for alpha in np.arange(0.0, 1.01, 0.1):
        blended = alpha * boost_val + (1 - alpha) * dl_val
        offsets = optimize_offsets(blended, y_val)
        preds = apply_offsets(blended, offsets)
        score = qwk(preds, y_val)
        marker = " <--" if score > best_score else ""
        print(f"{alpha:6.1f}  {score:20.4f}{marker}")

        alpha_dir = os.path.join(run_dir, f"alpha_{alpha:.1f}")
        os.makedirs(alpha_dir, exist_ok=True)
        save_confusion_matrix_plot(
            alpha_dir, preds, y_val,
            filename="confusion_matrix.png",
            title=f"Ensemble alpha={alpha:.1f} QWK={score:.4f}",
        )
        save_per_class_accuracy_plot(
            alpha_dir, preds, y_val,
            filename="per_class_accuracy.png",
            title=f"Per-class accuracy - alpha={alpha:.1f} QWK={score:.4f}",
        )
        alpha_results.append({"alpha": round(float(alpha), 1), "qwk": round(score, 6)})

        if score > best_score:
            best_score, best_alpha = score, alpha

    print(f"\nBest: alpha={best_alpha:.1f}  QWK={best_score:.4f}")
    print(f"(alpha=1.0 -> pure boosting, alpha=0.0 -> pure DL)")

    summary = {
        "boost_model": BOOST,
        "dl_run": DL_RUN,
        "best_alpha": round(float(best_alpha), 1),
        "best_qwk": round(best_score, 6),
        "boost_val_qwk_raw": round(float(qwk(boost_val, y_val)), 6),
        "dl_val_qwk_raw": round(float(qwk(dl_val, y_val)), 6),
        "alpha_results": alpha_results,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Generate test submission with best blend
    blended_val = best_alpha * boost_val + (1 - best_alpha) * dl_val
    blended_test = best_alpha * boost_test + (1 - best_alpha) * dl_test
    final_offsets = optimize_offsets(blended_val, y_val)
    final_preds = apply_offsets(blended_test, final_offsets)

    submission = pd.DataFrame({"Id": test_ids, "Response": final_preds})
    submission.to_csv(OUT, index=False)
    print(f"\nSubmission saved: {OUT}  ({len(submission)} rows)")


if __name__ == "__main__":
    main()
