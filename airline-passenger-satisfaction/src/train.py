"""Logique d'entraînement : recherche d'hyperparamètres et comparaison des prétraitements."""

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from src.model import build_pipeline, param_grid


def make_cv(config):
    """Validation croisée stratifiée : chaque pli garde la proportion de satisfaits."""
    return StratifiedKFold(n_splits=config["evaluation"]["cv_folds"],
                           shuffle=True, random_state=config["seed"])


def tune_models(X, y, config):
    """Recherche les meilleurs hyperparamètres de chaque modèle activé.

    Args:
        X, y: données d'entraînement (train.csv uniquement).
        config: configuration complète.

    Returns:
        (searches, summary) : dict nom -> GridSearchCV ajusté, et DataFrame
        des scores moyens de validation croisée du meilleur réglage de chaque modèle.
    """
    evaluation = config["evaluation"]
    searches, rows = {}, []
    for name, model_config in config["models"].items():
        if not model_config.get("enabled", True):
            continue
        print(f"Recherche d'hyperparamètres : {name}")
        search = GridSearchCV(
            build_pipeline(name, config["features"], config["preprocessing"], config["seed"]),
            param_grid(model_config),
            scoring=evaluation["metrics"],
            refit=evaluation["refit_metric"],
            cv=make_cv(config),
            n_jobs=-1,
            return_train_score=True,
        )
        search.fit(X, y)
        searches[name] = search

        best = search.best_index_
        row = {"model": name, "best_params": _short_params(search.best_params_)}
        for metric in evaluation["metrics"]:
            row[f"cv_{metric}"] = search.cv_results_[f"mean_test_{metric}"][best]
        row["train_" + evaluation["refit_metric"]] = (
            search.cv_results_[f"mean_train_{evaluation['refit_metric']}"][best])
        rows.append(row)
    return searches, pd.DataFrame(rows).set_index("model")


def select_best(searches):
    """Retourne le nom et la recherche du modèle au meilleur score de validation croisée."""
    name = max(searches, key=lambda n: searches[n].best_score_)
    return name, searches[name]


def compare_preprocessing(X, y, config, model_params):
    """Évalue chaque variante de prétraitement avec les mêmes plis et les mêmes modèles.

    Les hyperparamètres de chaque modèle sont fixés (ceux trouvés par
    `tune_models`) pour isoler l'effet du prétraitement.

    Args:
        model_params: dict nom du modèle -> hyperparamètres (préfixés classifier__).

    Returns:
        DataFrame : une ligne par (variante, modèle), scores moyens et écarts-types.
    """
    metrics = config["evaluation"]["metrics"]
    rows = []
    for variant, overrides in config["experiments"].items():
        options = {**config["preprocessing"], **(overrides or {})}
        for name, params in model_params.items():
            print(f"Variante {variant} – {name}")
            pipeline = build_pipeline(name, config["features"], options, config["seed"])
            pipeline.set_params(**params)
            scores = cross_validate(pipeline, X, y, cv=make_cv(config),
                                    scoring=metrics, n_jobs=-1)
            row = {"variant": variant, "model": name}
            for metric in metrics:
                row[metric] = scores[f"test_{metric}"].mean()
                row[f"{metric}_std"] = scores[f"test_{metric}"].std()
            rows.append(row)
    return pd.DataFrame(rows).set_index(["variant", "model"])


def _short_params(params):
    """Retire le préfixe « classifier__ » pour un affichage lisible."""
    return {key.replace("classifier__", ""): value for key, value in params.items()}
