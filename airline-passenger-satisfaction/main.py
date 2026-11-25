"""Point d'entrée de l'entraînement.

Exécute tout le pipeline : chargement, recherche d'hyperparamètres en
validation croisée sur train.csv, choix du meilleur modèle, sauvegarde, puis
une seule évaluation finale sur test.csv avec interprétation.

Usage :
    python main.py                 # entraînement + évaluation
    python main.py --experiments   # ajoute la comparaison des prétraitements
"""

import argparse

import sklearn

from src.data import load_dataset, split_features_target
from src.evaluate import (evaluate_on_test, logistic_coefficients,
                          permutation_importances, satisfaction_rates)
from src.train import compare_preprocessing, select_best, tune_models
from src.utils import load_config, save_json, save_model, save_table


def parse_args():
    parser = argparse.ArgumentParser(description="Entraîne le modèle de satisfaction passagers.")
    parser.add_argument("--config", default="configs/config.yaml", help="fichier de configuration")
    parser.add_argument("--experiments", action="store_true",
                        help="comparer les variantes de prétraitement définies dans la config")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    paths, data_config = config["paths"], config["data"]
    results = paths["results_dir"]

    # 1. Données d'entraînement (test.csv n'est lu qu'à l'étape 4).
    train_df = load_dataset(paths["train_csv"])
    X_train, y_train = split_features_target(train_df, data_config)
    print(f"Entraînement : {X_train.shape[0]} passagers, {X_train.shape[1]} variables")

    # 2. Recherche d'hyperparamètres et choix du modèle en validation croisée.
    searches, summary = tune_models(X_train, y_train, config)
    save_table(summary, f"{results}/model_comparison.csv")
    print(summary.round(4).to_string())

    if args.experiments:
        best_params = {name: search.best_params_ for name, search in searches.items()}
        comparison = compare_preprocessing(X_train, y_train, config, best_params)
        save_table(comparison, f"{results}/preprocessing_comparison.csv")
        print(comparison.round(4).to_string())

    # 3. Modèle retenu, déjà réentraîné sur tout train.csv par GridSearchCV.
    best_name, best_search = select_best(searches)
    model = best_search.best_estimator_
    save_model(model, paths["model"])
    print(f"Modèle retenu : {best_name} {best_search.best_params_}")

    # 4. Évaluation finale, une seule fois, sur test.csv.
    X_test, y_test = split_features_target(load_dataset(paths["test_csv"]), data_config)
    metrics, confusion = evaluate_on_test(model, X_test, y_test)
    save_json({"model": best_name,
               "params": {k.replace("classifier__", ""): v
                          for k, v in best_search.best_params_.items()},
               "test_metrics": metrics,
               "sklearn_version": sklearn.__version__},
              f"{results}/test_metrics.json")
    save_table(confusion, f"{results}/confusion_matrix.csv")
    print({k: round(v, 4) for k, v in metrics.items()})
    print(confusion.to_string())

    # 5. Interprétation.
    importances = permutation_importances(model, X_test, y_test,
                                          config["evaluation"]["permutation_repeats"],
                                          config["seed"])
    save_table(importances, f"{results}/permutation_importance.csv")
    print(importances.head(10).round(4).to_string())

    if "logistic_regression" in searches:
        coefficients = logistic_coefficients(searches["logistic_regression"].best_estimator_)
        save_table(coefficients, f"{results}/logistic_coefficients.csv")

    rates = satisfaction_rates(train_df, data_config["target"], data_config["positive_label"],
                               ["Online boarding", "Inflight wifi service",
                                "Type of Travel", "Class", "Customer Type"])
    save_table(rates, f"{results}/satisfaction_rates.csv")
    print(f"Modèle sauvegardé dans {paths['model']}, résultats dans {results}/")


if __name__ == "__main__":
    main()
