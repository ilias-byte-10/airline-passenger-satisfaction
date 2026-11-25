"""Inférence avec le modèle entraîné, indépendante de l'entraînement.

Charge models/best_model.pkl et prédit la satisfaction de nouveaux passagers.

Usage :
    python test.py                       # trois passagers d'exemple
    python test.py --input passagers.csv # fichier au même format que train.csv
"""

import argparse

import pandas as pd

from src.data import load_dataset
from src.utils import load_config, load_model

# Passagers fictifs, au format brut des fichiers Kaggle.
EXAMPLE_PASSENGERS = pd.DataFrame([
    {   # Voyage d'affaires en Business, services bien notés.
        "Gender": "Female", "Customer Type": "Loyal Customer", "Age": 41,
        "Type of Travel": "Business travel", "Class": "Business", "Flight Distance": 2450,
        "Inflight wifi service": 4, "Departure/Arrival time convenient": 4,
        "Ease of Online booking": 4, "Gate location": 3, "Food and drink": 4,
        "Online boarding": 5, "Seat comfort": 5, "Inflight entertainment": 5,
        "On-board service": 5, "Leg room service": 4, "Baggage handling": 5,
        "Checkin service": 4, "Inflight service": 5, "Cleanliness": 4,
        "Departure Delay in Minutes": 0, "Arrival Delay in Minutes": 0,
    },
    {   # Voyage personnel en Eco, embarquement et Wi-Fi mal notés, retard.
        "Gender": "Male", "Customer Type": "disloyal Customer", "Age": 23,
        "Type of Travel": "Personal Travel", "Class": "Eco", "Flight Distance": 520,
        "Inflight wifi service": 2, "Departure/Arrival time convenient": 3,
        "Ease of Online booking": 2, "Gate location": 3, "Food and drink": 2,
        "Online boarding": 2, "Seat comfort": 3, "Inflight entertainment": 2,
        "On-board service": 3, "Leg room service": 2, "Baggage handling": 3,
        "Checkin service": 3, "Inflight service": 3, "Cleanliness": 2,
        "Departure Delay in Minutes": 45, "Arrival Delay in Minutes": 50,
    },
    {   # Cas intermédiaire : Eco Plus, notes moyennes, retard d'arrivée inconnu.
        "Gender": "Female", "Customer Type": "Loyal Customer", "Age": 35,
        "Type of Travel": "Business travel", "Class": "Eco Plus", "Flight Distance": 1100,
        "Inflight wifi service": 3, "Departure/Arrival time convenient": 3,
        "Ease of Online booking": 3, "Gate location": 2, "Food and drink": 3,
        "Online boarding": 4, "Seat comfort": 3, "Inflight entertainment": 3,
        "On-board service": 4, "Leg room service": 3, "Baggage handling": 4,
        "Checkin service": 3, "Inflight service": 4, "Cleanliness": 3,
        "Departure Delay in Minutes": 10, "Arrival Delay in Minutes": None,
    },
])


def parse_args():
    parser = argparse.ArgumentParser(description="Prédit la satisfaction de nouveaux passagers.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", help="CSV de passagers (sinon exemples intégrés)")
    return parser.parse_args()


def predict(model, passengers):
    """Retourne les prédictions et la probabilité d'être satisfait.

    Args:
        model: Pipeline entraîné (prétraitement + classifieur).
        passengers: DataFrame aux colonnes brutes du jeu de données.
    """
    probability = model.predict_proba(passengers)[:, 1]
    label = ["satisfied" if p >= 0.5 else "neutral or dissatisfied" for p in probability]
    return pd.DataFrame({"prediction": label, "proba_satisfied": probability.round(3)},
                        index=passengers.index)


def main():
    args = parse_args()
    config = load_config(args.config)
    model = load_model(config["paths"]["model"])

    passengers = load_dataset(args.input) if args.input else EXAMPLE_PASSENGERS
    # Les colonnes en trop (id, cible…) sont ignorées par le prétraitement.
    print(predict(model, passengers).to_string())


if __name__ == "__main__":
    main()
