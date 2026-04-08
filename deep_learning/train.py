from __future__ import annotations
import argparse
import os
import sys
from sklearn.model_selection import train_test_split
from config import DataConfig, MODEL_CONFIGS
from logging_utils import setup_logging, get_logger
from metrics import optimize_offsets, qwk as _qwk, apply_offsets as _apply
from models import MODEL_REGISTRY
from preprocessing import load_data
from trainer import run_training, generate_submission

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraînement d'un modèle de deep learning tabulaire sur le dataset Prudential."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Architecture du modèle à entraîner.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_cfg = DataConfig()
    model_cfg = MODEL_CONFIGS[args.model]

    setup_logging(log_dir=data_cfg.log_dir, model_name=args.model)
    logger = get_logger(__name__)

    logger.info(f"{'=' * 50}")
    logger.info(f"Modèle : {args.model}")
    logger.info(f"Config : {model_cfg}")
    logger.info(f"{'=' * 50}")

    # 1. Chargement des données
    logger.info("Chargement des données...")
    X_all, y_all, X_test, ids_test = load_data(
        train_raw=data_cfg.train_raw,
        test_raw=data_cfg.test_raw,
        train_clean=data_cfg.train_clean,
        test_clean=data_cfg.test_clean,
        use_cached=data_cfg.use_cached,
        random_state=data_cfg.random_state,
    )
    logger.info(f"Entraînement : {X_all.shape} | Test : {X_test.shape}")
    input_dim = X_all.shape[1]

    # 2. Séparation train / validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=data_cfg.val_size,
        random_state=data_cfg.random_state,
        stratify=y_all.astype(int),
    )
    logger.info(f"Découpage — train : {X_train.shape[0]:,} | val : {X_val.shape[0]:,}")

    # 3. Entraînement et évaluation sur la validation
    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(input_dim=input_dim, config=model_cfg)
    qwk_raw, qwk_offset, offsets = run_training(model, X_train, y_train, X_val, y_val)

    # 4. Réentraînement sur l'ensemble complet et génération de la soumission
    submission_path = os.path.join(data_cfg.output_dir, f"submission_dl_{args.model}.csv")

    if data_cfg.retrain:
        logger.info("Réentraînement sur l'ensemble des données d'entraînement...")
        full_model = model_cls(input_dim=input_dim, config=model_cfg)
        full_model.fit(X_all, y_all, X_val, y_val)

        val_preds_full = full_model.predict(X_val)
        offsets_full = optimize_offsets(val_preds_full, y_val)
        qwk_raw_full = _qwk(val_preds_full, y_val)
        qwk_offset_full = _qwk(_apply(val_preds_full, offsets_full), y_val)

        generate_submission(full_model, X_test, ids_test, offsets_full, submission_path)
    else:
        logger.info("Réentraînement désactivé (data_cfg.retrain=False).")
        qwk_raw_full = qwk_raw
        qwk_offset_full = qwk_offset
        offsets_full = offsets
        generate_submission(model, X_test, ids_test, offsets, submission_path)

    # 5. Résumé final
    BOOSTING_BASELINE = 0.6544
    delta = qwk_offset_full - BOOSTING_BASELINE
    delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"

    logger.info("")
    logger.info(f"{'=' * 50}")
    logger.info(f"  RÉSUMÉ FINAL — {args.model}")
    logger.info(f"{'=' * 50}")
    logger.info(f"  {'Métrique':<30} {'Valeur':>8}")
    logger.info(f"  {'-' * 39}")
    logger.info(f"  {'QWK val (brut, split 80%)':<30} {qwk_raw:>8.4f}")
    logger.info(f"  {'QWK val (offsets, split 80%)':<30} {qwk_offset:>8.4f}")
    if data_cfg.retrain:
        logger.info(f"  {'QWK val (brut, réentraîn.)':<30} {qwk_raw_full:>8.4f}")
        logger.info(f"  {'QWK val (offsets, réentraîn.)':<30} {qwk_offset_full:>8.4f}")
    logger.info(f"  {'-' * 39}")
    logger.info(f"  {'Baseline boosting (CatBoost)':<30} {BOOSTING_BASELINE:>8.4f}")
    logger.info(f"  {'Écart vs baseline':<30} {delta_str:>8}")
    logger.info(f"{'=' * 50}")
    logger.info(f"  Soumission : {submission_path}")
    logger.info(f"{'=' * 50}")
    logger.info("")
    logger.info("Terminé.")


if __name__ == "__main__":
    main()
