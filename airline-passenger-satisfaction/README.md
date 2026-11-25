# Prédiction de la satisfaction des passagers aériens

Projet intégré de Machine Learning – UIR, année 2026/2027 (Sujet 12).
ELHATTAB Mohammed, SALHI Haitem, MOUSSAOUI Ilias, FAQDANE Sami Saif Islam.

Classification binaire : prédire si un passager est **satisfait** ou **neutre/insatisfait**
à partir de son profil, de son voyage et des notes qu'il donne aux services, puis
identifier les services les plus liés à la satisfaction.

## Structure

```
main.py                  # entraînement complet + évaluation + sauvegarde
test.py                  # inférence indépendante avec le modèle sauvegardé
configs/config.yaml      # chemins, colonnes, prétraitement, grilles d'hyperparamètres
src/data.py              # chargement et séparation X / y
src/preprocessing.py     # pipeline de prétraitement (imputation, encodage, standardisation)
src/model.py             # modèles et assemblage Pipeline
src/train.py             # GridSearchCV et comparaison des prétraitements
src/evaluate.py          # métriques, matrice de confusion, interprétation
src/utils.py             # configuration, sauvegarde/chargement
models/best_model.pkl    # modèle entraîné (Pipeline complet)
results/                 # tableaux produits par main.py
```

## Installation

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate    macOS/Linux : source .venv/bin/activate
pip install -r requirements.txt
```

Télécharger `train.csv` et `test.csv` depuis
[Kaggle – Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
et les placer dans `data/`.

## Entraîner

```bash
python main.py                 # ~1 min
python main.py --experiments   # + comparaison des variantes de prétraitement (~2 min)
```

Le script cherche les hyperparamètres de chaque modèle en validation croisée stratifiée
à 5 plis sur `train.csv`, retient le meilleur selon le F1, le sauvegarde dans
`models/best_model.pkl`, puis l'évalue **une seule fois** sur `test.csv`.

## Tester sur de nouveaux passagers

```bash
python test.py                          # trois passagers d'exemple
python test.py --input data/test.csv    # n'importe quel CSV au format Kaggle
```

## Méthode

- **Prétraitement** : médiane pour les 310 retards manquants, one-hot encoding des
  4 variables catégorielles, standardisation des variables numériques. Tout est
  ajusté dans un `Pipeline` scikit-learn, donc uniquement sur les plis d'entraînement.
- **Notes à 0** : 0 signifie « non applicable » (environ 5 300 passagers pour l'horaire,
  4 500 pour la réservation en ligne, 3 100 pour le Wi-Fi). Une option ajoute un
  indicateur dédié.
- **Modèles** : arbre de décision et régression logistique (forêt aléatoire disponible,
  désactivée dans la config).
- **Évaluation** : accuracy, précision, rappel, F1, ROC-AUC, matrice de confusion.
- **Interprétation** : importance par permutation sur le test, coefficients de la
  régression logistique, taux de satisfaction par modalité.

## Résultats principaux

| Modèle (CV 5 plis sur train) | Meilleurs réglages | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| Arbre de décision | gini, max_depth=15, min_samples_leaf=5 | 0,952 | 0,944 | 0,982 |
| Régression logistique | C=0,1 | 0,875 | 0,853 | 0,927 |

Sur `test.csv`, l'arbre retenu obtient **95,5 % d'accuracy**, F1 = 0,948, ROC-AUC = 0,983.

Effet du prétraitement (détails dans `results/preprocessing_comparison.csv`) :
- l'indicateur « note = 0 non applicable » fait passer la régression logistique de
  0,875 à 0,903 d'accuracy, mais ne change rien pour l'arbre, qui isole déjà la valeur 0
  par un seuil ;
- les options sur les retards (indicateur, log, suppression du retard au départ) ont
  un effet négligeable ;
- sans les notes de service (scénario « avant le vol »), l'accuracy tombe à environ 0,79.
