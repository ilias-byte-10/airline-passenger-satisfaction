"""Fonctions utilitaires : configuration, sauvegarde et chargement."""

import json
from pathlib import Path

import joblib
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path="configs/config.yaml"):
    """Charge la configuration YAML.

    Args:
        path: chemin du fichier, relatif à la racine du projet ou absolu.

    Returns:
        dict contenant la configuration.
    """
    with open(resolve_path(path), encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(path):
    """Rend un chemin relatif à la racine du projet, quel que soit le dossier courant."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def save_model(model, path):
    """Sauvegarde un modèle entraîné avec joblib."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path):
    """Charge un modèle sauvegardé avec `save_model`."""
    return joblib.load(resolve_path(path))


def save_json(data, path):
    """Écrit un dictionnaire en JSON lisible."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    return path


def save_table(table, path):
    """Écrit un DataFrame en CSV."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=True)
    return path
