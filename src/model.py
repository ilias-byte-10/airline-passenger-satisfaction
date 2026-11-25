"""Définition des modèles et de leurs espaces de recherche."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_preprocessor


def make_estimator(name, seed):
    """Crée un classifieur non entraîné à partir de son nom.

    Args:
        name: « decision_tree », « logistic_regression » ou « random_forest ».
        seed: graine aléatoire pour la reproductibilité.
    """
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed)
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1)
    raise ValueError(f"Modèle inconnu : {name}")


def build_pipeline(name, features, preprocessing_options, seed):
    """Assemble prétraitement et classifieur dans un seul Pipeline.

    Le Pipeline complet est ce qui est sauvegardé : à l'inférence, il suffit de
    lui donner les colonnes brutes d'un passager.
    """
    return Pipeline([
        ("preprocess", build_preprocessor(features, preprocessing_options)),
        ("classifier", make_estimator(name, seed)),
    ])


def param_grid(model_config):
    """Préfixe les hyperparamètres de la config pour GridSearchCV."""
    return {f"classifier__{key}": values for key, values in model_config["params"].items()}
