"""
Inférence churn — reconstruit les features brutes attendues par le preprocessor
puis renvoie la probabilité de churn.

Ce module est la SOURCE UNIQUE de vérité pour la logique de feature engineering
en production. Elle réplique EXACTEMENT les notebooks 03 (feature engineering)
et 05 (encodage avant preprocessor) :

  - gender : "Male" -> 1, "Female" -> 0                          (nb 05)
  - service_cols : "Yes" -> 1 sinon 0                            (nb 03)
  - IsMonthly = (Contract == "Monthly")                          (nb 03)
  - IsSingle  = (Partner == 1) & (Dependents == 1)               (nb 03)
  - TotalServices = somme des 6 services                         (nb 03)
  - AvgMonthlyCharge = TotalCharges / tenure                     (nb 03)
  - ChargeDiff = MonthlyCharges - AvgMonthlyCharge               (nb 03)
  - TenureGroups = pd.cut(tenure, [0,12,24,48,72,100], ...)      (nb 03)

Le ColumnTransformer (preprocessor.pkl) attend ces 25 colonnes brutes (ordre
libre, il sélectionne par nom) et produit 29 features. best_model.pkl prédit.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Bornes/labels identiques au notebook 03 (tenure observé : 1..72)
TENURE_BINS = [0, 12, 24, 48, 72, 100]
TENURE_LABELS = ["0-1", "1-2", "2-4", "4-6", "Over 6 years"]

# Valeurs autorisées pour les champs catégoriels (pour construire les widgets)
CHOICES = {
    "gender": ["Female", "Male"],
    "Contract": ["Monthly", "One year", "Two year"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    "yes_no": ["No", "Yes"],
}

# Colonnes brutes attendues par le preprocessor (= feature_names_in_)
RAW_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "TenureGroups", "IsMonthly",
    "IsSingle", "TotalServices", "AvgMonthlyCharge", "ChargeDiff",
]


def _yes(v) -> int:
    """Normalise une saisie en 0/1 ('Yes'/True/1 -> 1)."""
    if isinstance(v, str):
        return 1 if v.strip().lower() == "yes" else 0
    return int(bool(v))


def build_raw_row(inp: dict) -> pd.DataFrame:
    """
    Construit une ligne brute (DataFrame 1x25) à partir d'un dictionnaire
    de saisies lisibles. Recalcule toutes les features dérivées.

    Clés attendues dans `inp` :
        gender (Female/Male), SeniorCitizen (0/1), Partner (Yes/No),
        Dependents (Yes/No), tenure (int 1..72), PhoneService (Yes/No),
        MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
        DeviceProtection, TechSupport, StreamingTV, StreamingMovies (Yes/No),
        Contract, PaperlessBilling (Yes/No), PaymentMethod,
        MonthlyCharges (float), TotalCharges (float)
    """
    tenure = max(int(inp["tenure"]), 1)  # éviter division par zéro
    monthly = float(inp["MonthlyCharges"])
    total = float(inp["TotalCharges"])

    services = {c: _yes(inp[c]) for c in SERVICE_COLS}
    total_services = sum(services.values())

    contract = inp["Contract"]
    partner = _yes(inp["Partner"])
    dependents = _yes(inp["Dependents"])

    avg_monthly = total / tenure
    charge_diff = monthly - avg_monthly

    tenure_group = pd.cut(
        pd.Series([tenure]),
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        include_lowest=True,
    ).iloc[0]

    row = {
        "gender": 1 if inp["gender"] == "Male" else 0,
        "SeniorCitizen": int(inp["SeniorCitizen"]),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": float(tenure),
        "PhoneService": _yes(inp["PhoneService"]),
        "MultipleLines": inp["MultipleLines"],
        "InternetService": inp["InternetService"],
        **services,
        "Contract": contract,
        "PaperlessBilling": _yes(inp["PaperlessBilling"]),
        "PaymentMethod": inp["PaymentMethod"],
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "TenureGroups": str(tenure_group),
        "IsMonthly": 1 if contract == "Monthly" else 0,
        "IsSingle": 1 if (partner == 1 and dependents == 1) else 0,
        "TotalServices": total_services,
        "AvgMonthlyCharge": avg_monthly,
        "ChargeDiff": charge_diff,
    }
    return pd.DataFrame([row])[RAW_COLUMNS]


_preprocessor = None
_model = None


def load_artifacts(models_dir: Path = MODELS):
    """Charge (et met en cache) le preprocessor et le meilleur modèle."""
    global _preprocessor, _model
    if _preprocessor is None:
        _preprocessor = joblib.load(models_dir / "preprocessor.pkl")
    if _model is None:
        _model = joblib.load(models_dir / "best_model.pkl")
    return _preprocessor, _model


def predict_proba(inp: dict, threshold: float = 0.5) -> dict:
    """
    Renvoie la probabilité de churn et la décision pour un client.

    threshold : seuil de décision (ajustable selon le coût métier).
    """
    preprocessor, model = load_artifacts()
    raw = build_raw_row(inp)
    X = preprocessor.transform(raw)
    proba = float(model.predict_proba(X)[0, 1])
    return {
        "churn_probability": proba,
        "prediction": int(proba >= threshold),
        "threshold": threshold,
    }


if __name__ == "__main__":
    # Sanity check rapide
    sample = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Monthly", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0,
        "TotalCharges": 170.0,
    }
    print("Client à risque (mensuel) :", predict_proba(sample))
    sample2 = {**sample, "Contract": "Two year", "tenure": 60,
               "TotalCharges": 5000.0, "MonthlyCharges": 83.0}
    print("Client fidèle (2 ans)     :", predict_proba(sample2))
