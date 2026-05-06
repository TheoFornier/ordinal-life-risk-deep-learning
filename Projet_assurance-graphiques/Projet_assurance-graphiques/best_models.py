"""
GLO-7030 -- Extraction automatique des meilleurs modeles
========================================================
Interroge l'API Weights & Biases, isole le meilleur run
par type de modele et exporte le resultat dans best_models.json.

Usage:
    python best_models.py           # mode normal
    python best_models.py --debug   # mode diagnostic
"""

import sys
import math
import json
import wandb


PROJECT = "GLO7030-Prudential"
# Le chemin complet (FULL_PATH) sera généré dynamiquement dans main() avec ton compte actuel


def extract_config_value(val):
    """Deplie les valeurs de config W&B qui peuvent etre imbriquees."""
    if isinstance(val, dict) and "value" in val:
        return val["value"]
    return val


def detect_model_type(config, summary, run_name, tags):
    """Determine le type de modele depuis la config, le summary, le nom ou les tags."""
    # 1. Config directe
    for key in ["model_type", "model", "architecture"]:
        raw = config.get(key)
        val = extract_config_value(raw)
        if val and isinstance(val, str):
            return val

    # 2. Indices dans les metriques du summary
    if "catboost_best_iteration" in summary:
        return "CatBoost"
    if "round" in summary and "train_rmse" in summary:
        return "XGBoost"

    # 3. Nom du run
    name = run_name.lower()
    mappings = {
        "xgboost": "XGBoost", "xgb": "XGBoost",
        "catboost": "CatBoost", "cat": "CatBoost",
        "resnet": "ResNet", "tabpfn": "TabPFN",
        "tabm": "TabM", "mlp": "MLP", "transformer": "Transformer",
    }
    for key, label in mappings.items():
        if key in name:
            return label

    # 4. Tags
    for tag in (tags or []):
        for key, label in mappings.items():
            if key in tag.lower():
                return label

    # 5. Déduction via les hyperparamètres du sweep
    if "min_child_weight" in config or "colsample_bytree" in config:
        return "XGBoost"
    if "depth" in config and "iterations" in config:
        return "CatBoost"

    return "Gábor_Model"


def extract_best_qwk(summary):
    """Recupere le meilleur score QWK depuis le summary d'un run."""
    for key in ["best_qwk", "val_qwk_offset", "val_qwk_simple"]:
        val = summary.get(key)
        if val is not None:
            try:
                val = float(val)
                if not math.isnan(val):
                    return val
            except (TypeError, ValueError):
                continue
    return None


def clean_hyperparams(config):
    """Extrait les hyperparametres utiles depuis la config brute."""
    ignore = {"model_type", "clean_method", "use_offsets", "_wandb"}
    result = {}
    for k, v in config.items():
        if k.startswith("_") or k in ignore:
            continue
        result[k] = extract_config_value(v)
    return result


def debug_mode(api, full_path):
    """Affiche les informations brutes de tous les runs pour diagnostic."""
    runs = api.runs(full_path)
    print(f"\n{'='*60}")
    print(f"MODE DIAGNOSTIC - {len(runs)} runs trouves")
    print(f"Chemin utilise : {full_path}")
    print(f"{'='*60}\n")

    # Compteurs par etat
    states = {}
    has_config = 0
    has_qwk = 0

    for run in runs:
        states[run.state] = states.get(run.state, 0) + 1
        cfg = dict(run.config)
        if cfg:
            has_config += 1
        summary_clean = {k: v for k, v in dict(run.summary).items() if not k.startswith("_")}
        qwk_keys = [k for k in summary_clean if "qwk" in k.lower()]
        if qwk_keys:
            has_qwk += 1

        # Afficher uniquement les runs qui ont des donnees utiles
        if cfg or qwk_keys:
            print(f"--- {run.name} (id: {run.id}, state: {run.state}) ---")
            print(f"  Config: {cfg}")
            print(f"  Metriques: {summary_clean}")
            print(f"  Tags: {run.tags}")
            print()

    print(f"{'='*60}")
    print(f"RESUME :")
    print(f"  Etats      : {states}")
    print(f"  Avec config: {has_config}/{len(runs)}")
    print(f"  Avec QWK   : {has_qwk}/{len(runs)}")
    print(f"{'='*60}")


def main():
    api = wandb.Api()
    entity = api.default_entity
    full_path = f"{entity}/{PROJECT}"

    print(f"Connexion a l'API Weights & Biases...")
    print(f"  Compte detecte : {entity}")
    print(f"  Chemin : {full_path}")

    if "--debug" in sys.argv:
        debug_mode(api, full_path)
        return

    try:
        runs = api.runs(full_path)
    except Exception as e:
        print(f"[ERREUR] Impossible de se connecter a W&B: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"{len(runs)} runs trouves. Analyse en cours...")

    best_runs = {}
    skipped = 0

    for run in runs:
        if run.state != "finished":
            skipped += 1
            continue

        cfg = dict(run.config)
        sumry = dict(run.summary)

        model_type = detect_model_type(cfg, sumry, run.name, run.tags)
        score = extract_best_qwk(sumry)

        if score is None:
            skipped += 1
            continue

        if model_type not in best_runs or score > best_runs[model_type]["score"]:
            best_runs[model_type] = {"score": score, "run": run, "config": cfg, "summary": sumry}

    print(f"  {len(best_runs)} types de modeles detectes, {skipped} runs ignores.")

    if not best_runs:
        print("[AVERTISSEMENT] Aucun run exploitable n'a ete trouve.")
        print("Lancez 'python best_models.py --debug' pour diagnostiquer.")
        sys.exit(0)

    palette = [
        "#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6",
        "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
        "#14b8a6", "#e11d48", "#a855f7", "#0ea5e9", "#d97706",
    ]

    models_data = []

    api = wandb.Api() # on s'assure d'avoir l'api dispo ici
    for i, (model_type, data) in enumerate(sorted(best_runs.items())):
        run = data["run"]
        cfg = data["config"]
        sumry = data["summary"]

        # -------------------------------------------------------------
        # FIX : Si la config est vide (bug API W&B), on recharge 
        # le run complet individuellement pour forcer le téléchargement.
        # -------------------------------------------------------------
        if not cfg:
            print(f"  - Re-telechargement de la config pour {model_type} ({run.id})...")
            try:
                full_run = api.run(f"{full_path}/{run.id}")
                cfg = dict(full_run.config)
            except Exception:
                pass

        author = "Inconnu"
        try:
            author = run.user.username if run.user else "Inconnu"
        except Exception:
            pass

        # Telechargement des VRAIES courbes d'apprentissage depuis W&B
        print(f"  - Telechargement de l'historique pour {model_type} ({run.id})...")
        hist = run.history(keys=["train_rmse", "val_rmse"], samples=2000)
        train_curve = hist["train_rmse"].dropna().tolist() if "train_rmse" in hist else None
        val_curve = hist["val_rmse"].dropna().tolist() if "val_rmse" in hist else None
        
        # Si W&B ne renvoie rien, on remet a None
        if train_curve and len(train_curve) == 0: train_curve = None
        if val_curve and len(val_curve) == 0: val_curve = None

        models_data.append({
            "name": model_type,
            "author": author,
            "color": palette[i % len(palette)],
            "run_id": run.id,
            "qwk_simple": sumry.get("val_qwk_simple"),
            "qwk_offset": sumry.get("val_qwk_offset"),
            "val_rmse": sumry.get("val_rmse"),
            "train_rmse": sumry.get("train_rmse"),
            "hyperparams": clean_hyperparams(cfg),
            "train_rmse_curve": train_curve,
            "val_rmse_curve": val_curve,
            "confusion_matrix": None,
            "pred_distribution": None,
            "true_distribution": None,
        })

    output_file = "best_models.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(models_data, f, indent=4)

    print(f"\n{'='*60}")
    print(f"EXTRACTION TERMINEE - {len(models_data)} modeles ecrits dans '{output_file}'")
    for m in models_data:
        best = m["qwk_offset"] if m["qwk_offset"] else m["qwk_simple"]
        print(f"  - {m['name']:15s}  QWK: {best:.4f}  (run: {m['run_id']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
