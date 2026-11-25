"""Évaluation sur le jeu de test et interprétation du modèle."""

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

CLASS_NAMES = ["neutral or dissatisfied", "satisfied"]


def evaluate_on_test(model, X_test, y_test):
    """Calcule les métriques globales et par classe sur le jeu de test.

    Returns:
        (metrics, confusion) : dict de métriques et DataFrame de la matrice
        de confusion (lignes = vraie classe, colonnes = prédiction).
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        # Rappel de la classe 0 : part des insatisfaits correctement repérés.
        "recall_dissatisfied": recall_score(y_test, predictions, pos_label=0),
    }
    confusion = pd.DataFrame(confusion_matrix(y_test, predictions),
                             index=[f"réel : {c}" for c in CLASS_NAMES],
                             columns=[f"prédit : {c}" for c in CLASS_NAMES])
    return metrics, confusion


def permutation_importances(model, X, y, repeats, seed, scoring="f1"):
    """Importance de chaque variable d'origine par permutation.

    On mélange une colonne à la fois et on mesure la baisse du score : plus la
    baisse est forte, plus le modèle dépend de cette variable. Contrairement à
    l'importance interne d'un arbre, cette mesure se calcule sur des données
    non vues et s'applique à n'importe quel modèle.

    Returns:
        DataFrame trié : baisse moyenne du score et écart-type.
    """
    result = permutation_importance(model, X, y, n_repeats=repeats,
                                    random_state=seed, scoring=scoring, n_jobs=-1)
    return (pd.DataFrame({"importance_mean": result.importances_mean,
                          "importance_std": result.importances_std}, index=X.columns)
            .sort_values("importance_mean", ascending=False))


def logistic_coefficients(pipeline):
    """Coefficients d'une régression logistique sur variables standardisées.

    Un coefficient positif augmente la probabilité d'être satisfait. Les
    variables étant standardisées, les valeurs absolues sont comparables.
    """
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefficients = pipeline.named_steps["classifier"].coef_[0]
    table = pd.DataFrame({"coefficient": coefficients}, index=names)
    return table.reindex(table["coefficient"].abs().sort_values(ascending=False).index)


def satisfaction_rates(df, target, positive_label, columns):
    """Taux de satisfaction observé par modalité de quelques variables.

    Ces taux décrivent des associations dans les données, pas des causes.

    Returns:
        DataFrame : variable, modalité, effectif et pourcentage de satisfaits.
    """
    satisfied = df[target].eq(positive_label)
    tables = []
    for column in columns:
        grouped = satisfied.groupby(df[column])
        table = pd.DataFrame({"passengers": grouped.size(),
                              "satisfied_pct": (grouped.mean() * 100).round(1)})
        table.index = pd.MultiIndex.from_product([[column], table.index],
                                                 names=["variable", "value"])
        tables.append(table)
    return pd.concat(tables)
