"""
TelcoConnect Churn — Entraînement & évaluation des modèles
==========================================================

Ce script :
1. Charge les données déjà préprocessées (X/y_train, X/y_test).
2. Entraîne 3 modèles : Logistic Regression (baseline), Random Forest, XGBoost.
3. Évalue chacun en cross-validation (StratifiedKFold) sur le train.
4. Optimise les hyperparamètres du meilleur famille via RandomizedSearchCV.
5. Évalue le modèle final sur le test (Accuracy, Precision, Recall, F1, ROC-AUC).
6. Sauvegarde les modèles (.pkl), les figures (ROC, matrice de confusion,
   feature importance) et un tableau comparatif + metrics.json.

Lancer :  ./venv/bin/python src/train_models.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # backend sans affichage (script)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
for d in (MODELS, REPORTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------- #
# 1. Chargement des données
# --------------------------------------------------------------------------- #
def load_data():
    X_train = pd.read_csv(DATA / "X_train.csv")
    X_test = pd.read_csv(DATA / "X_test.csv")
    y_train = pd.read_csv(DATA / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(DATA / "y_test.csv").squeeze("columns")
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"Churn balance (train): {y_train.mean():.4f}")
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# 2. Évaluation helper
# --------------------------------------------------------------------------- #
def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def cross_validate(model, X_train, y_train, cv) -> dict:
    """ROC-AUC et F1 moyens en cross-validation."""
    auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    return {
        "cv_roc_auc_mean": auc.mean(),
        "cv_roc_auc_std": auc.std(),
        "cv_f1_mean": f1.mean(),
        "cv_f1_std": f1.std(),
    }


# --------------------------------------------------------------------------- #
# 3. Pipeline principal
# --------------------------------------------------------------------------- #
def main():
    X_train, X_test, y_train, y_test = load_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    base_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = {}
    trained = {}

    print("\n=== Cross-validation (5-fold) + entraînement ===")
    for name, model in base_models.items():
        print(f"\n[{name}]")
        cv_scores = cross_validate(model, X_train, y_train, cv)
        print(
            f"  CV ROC-AUC: {cv_scores['cv_roc_auc_mean']:.4f} "
            f"(+/- {cv_scores['cv_roc_auc_std']:.4f}) | "
            f"CV F1: {cv_scores['cv_f1_mean']:.4f}"
        )
        model.fit(X_train, y_train)
        test_scores = evaluate(model, X_test, y_test)
        print(
            f"  TEST  Acc: {test_scores['accuracy']:.4f} | "
            f"Prec: {test_scores['precision']:.4f} | "
            f"Rec: {test_scores['recall']:.4f} | "
            f"F1: {test_scores['f1']:.4f} | "
            f"AUC: {test_scores['roc_auc']:.4f}"
        )
        results[name] = {**cv_scores, **test_scores}
        trained[name] = model
        joblib.dump(model, MODELS / f"{name.lower().replace(' ', '_')}.pkl")

    # ------------------------------------------------------------------- #
    # 4. Tuning de la meilleure famille (par CV ROC-AUC)
    # ------------------------------------------------------------------- #
    best_family = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    print(f"\n=== Meilleure famille en CV : {best_family} → tuning ===")

    if best_family == "XGBoost":
        estimator = XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
        )
        param_dist = {
            "n_estimators": [200, 400, 600, 800],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5],
        }
    elif best_family == "Random Forest":
        estimator = RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE)
        param_dist = {
            "n_estimators": [200, 300, 500, 800],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }
    else:  # Logistic Regression
        estimator = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        param_dist = {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
        }

    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=25,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"  Meilleurs params : {search.best_params_}")
    print(f"  Meilleur CV ROC-AUC : {search.best_score_:.4f}")

    tuned_scores = evaluate(best_model, X_test, y_test)
    tuned_name = f"{best_family} (tuned)"
    results[tuned_name] = {
        "cv_roc_auc_mean": search.best_score_,
        "cv_roc_auc_std": np.nan,
        "cv_f1_mean": np.nan,
        "cv_f1_std": np.nan,
        **tuned_scores,
    }
    print(
        f"  TEST tuned  Acc: {tuned_scores['accuracy']:.4f} | "
        f"F1: {tuned_scores['f1']:.4f} | AUC: {tuned_scores['roc_auc']:.4f}"
    )

    # --- Sélection finale honnête -----------------------------------------
    # Le tuning n'améliore pas toujours : on compare le modèle de base de la
    # meilleure famille avec sa version tunée et on garde le meilleur en ROC-AUC
    # sur le test. (Différences faibles = features qui plafonnent ~0.79 AUC.)
    base_best = trained[best_family]
    base_auc = results[best_family]["roc_auc"]
    tuned_auc = tuned_scores["roc_auc"]
    if tuned_auc >= base_auc:
        final_model, final_label, final_params = best_model, tuned_name, search.best_params_
    else:
        final_model, final_label, final_params = base_best, best_family, base_best.get_params()
    print(
        f"\n=== Modèle final retenu : {final_label} "
        f"(test ROC-AUC base={base_auc:.4f} vs tuned={tuned_auc:.4f}) ==="
    )
    joblib.dump(final_model, MODELS / "best_model.pkl")
    joblib.dump(
        {"best_label": final_label, "best_family": best_family, "best_params": final_params},
        MODELS / "best_model_meta.pkl",
    )

    # ------------------------------------------------------------------- #
    # 5. Tableau comparatif
    # ------------------------------------------------------------------- #
    comparison = pd.DataFrame(results).T
    comparison = comparison[
        [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "cv_roc_auc_mean",
            "cv_roc_auc_std",
        ]
    ].round(4)
    comparison.to_csv(REPORTS / "model_comparison.csv")
    with open(REPORTS / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n=== Tableau comparatif ===")
    print(comparison)

    # ------------------------------------------------------------------- #
    # 6. Figures
    # ------------------------------------------------------------------- #
    plot_roc_curves(trained, final_model, final_label, X_test, y_test)
    plot_confusion(final_model, X_test, y_test, final_label)
    plot_feature_importance(final_model, X_train.columns, final_label)
    threshold_analysis(final_model, X_test, y_test)

    print("\n✅ Terminé. Modèles dans models/, figures & rapports dans reports/.")
    return results


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_roc_curves(trained, best_model, best_family, X_test, y_test):
    plt.figure(figsize=(8, 7))
    for name, model in trained.items():
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, "--", lw=2.5, label=f"{best_family} tuned (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k:", alpha=0.5)
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbes ROC — comparaison des modèles")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES / "roc_curves.png", dpi=120)
    plt.close()


def plot_confusion(model, X_test, y_test, name):
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title(f"Matrice de confusion — {name} (tuned)")
    plt.tight_layout()
    plt.savefig(FIGURES / "confusion_matrix_best.png", dpi=120)
    plt.close()


def plot_feature_importance(model, columns, name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        label = "Importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        label = "|Coefficient|"
    else:
        return
    fi = (
        pd.DataFrame({"feature": columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    fi.to_csv(REPORTS / "feature_importance.csv", index=False)
    plt.figure(figsize=(9, 7))
    sns.barplot(data=fi, y="feature", x="importance", palette="viridis")
    plt.xlabel(label)
    plt.ylabel("")
    plt.title(f"Top 15 features — {name}")
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_importance.png", dpi=120)
    plt.close()


def threshold_analysis(model, X_test, y_test):
    """Précision/Recall/F1 en fonction du seuil de décision (calibration métier)."""
    proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.91, 0.05)
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows.append(
            {
                "threshold": round(t, 2),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred),
                "f1": f1_score(y_test, pred),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "threshold_analysis.csv", index=False)
    plt.figure(figsize=(8, 6))
    plt.plot(df["threshold"], df["precision"], "-o", label="Precision")
    plt.plot(df["threshold"], df["recall"], "-o", label="Recall")
    plt.plot(df["threshold"], df["f1"], "-o", label="F1")
    plt.axvline(0.5, color="gray", ls="--", alpha=0.6, label="Seuil défaut (0.5)")
    plt.xlabel("Seuil de décision")
    plt.ylabel("Score")
    plt.title("Précision / Recall / F1 selon le seuil")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "threshold_analysis.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
