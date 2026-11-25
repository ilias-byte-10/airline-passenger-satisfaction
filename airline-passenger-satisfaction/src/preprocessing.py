"""Pipeline de prétraitement et de représentation des données.

Toutes les étapes apprises (médiane, moyennes et écarts-types, catégories)
sont ajustées dans un `ColumnTransformer` scikit-learn. Placé dans un
`Pipeline` avec le modèle, il est réajusté sur chaque pli d'entraînement de la
validation croisée, ce qui évite toute fuite d'information vers la validation.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


class NotApplicableFlags(BaseEstimator, TransformerMixin):
    """Ajoute une colonne binaire « non applicable » pour certaines notes.

    Dans ce jeu de données, une note de 0 ne veut pas dire « très mauvais » :
    elle signifie que le service n'était pas concerné. La colonne ajoutée permet
    au modèle de traiter ce cas à part au lieu de le placer sous la note 1.

    Args:
        flagged_columns: noms des notes pour lesquelles ajouter un indicateur.
    """

    def __init__(self, flagged_columns=()):
        self.flagged_columns = flagged_columns

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.flagged_ = [c for c in self.flagged_columns if c in X.columns]
        return self

    def transform(self, X):
        X = X.copy()
        for column in self.flagged_:
            X[f"{column} (non applicable)"] = (X[column] == 0).astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        flags = [f"{c} (non applicable)" for c in self.flagged_]
        return np.asarray([*self.feature_names_in_, *flags], dtype=object)


def build_preprocessor(features, options):
    """Construit le prétraitement selon les options choisies.

    Args:
        features: section `features` de la configuration (listes de colonnes).
        options: dict d'options booléennes (section `preprocessing`).

    Returns:
        ColumnTransformer non ajusté qui produit une matrice numérique.
    """
    delay_columns = list(features["delays"])
    if options.get("drop_departure_delay"):
        delay_columns = [c for c in delay_columns if "Departure" not in c]

    delay_steps = []
    if options.get("log_delays"):
        # log1p laisse les NaN intacts : l'imputation qui suit les remplace.
        delay_steps.append(("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    delay_steps += [
        ("impute", SimpleImputer(strategy="median",
                                 add_indicator=bool(options.get("missing_indicator")))),
        ("scale", StandardScaler()),
    ]

    flagged = features["ratings_with_not_applicable"] if options.get("not_applicable_flags") else []
    rating_pipeline = Pipeline([
        ("not_applicable", NotApplicableFlags(flagged)),
        ("scale", StandardScaler()),
    ])

    transformers = [
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         list(features["categorical"])),
        ("numeric", StandardScaler(), list(features["numeric"])),
        ("delays", Pipeline(delay_steps), delay_columns),
    ]
    if not options.get("drop_service_ratings"):
        transformers.append(("ratings", rating_pipeline, list(features["ratings"])))

    # Les colonnes non listées (aucune normalement) sont écartées.
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
