"""Chargement des données et séparation des variables et de la cible."""

import pandas as pd

from src.utils import resolve_path


def load_dataset(path):
    """Lit un fichier CSV du jeu Airline Passenger Satisfaction.

    Args:
        path: chemin du CSV (train ou test).

    Returns:
        DataFrame brut, une ligne par passager.
    """
    return pd.read_csv(resolve_path(path))


def split_features_target(df, data_config):
    """Sépare les variables explicatives X et la cible binaire y.

    Les colonnes d'index et d'identifiant sont retirées car elles ne décrivent
    pas le passager. La cible vaut 1 pour « satisfied » et 0 sinon.

    Args:
        df: DataFrame brut.
        data_config: section `data` de la configuration.

    Returns:
        (X, y) : DataFrame des variables et Series d'entiers 0/1.
    """
    target = data_config["target"]
    X = df.drop(columns=[target, *data_config["drop_columns"]], errors="ignore")
    y = (df[target] == data_config["positive_label"]).astype(int)
    return X, y
