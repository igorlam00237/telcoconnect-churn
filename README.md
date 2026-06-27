# Prédiction du Churn Clients — TelcoConnect 📊

Projet Data Science **de bout en bout** : prédire quels clients risquent de
résilier leur abonnement chez TelcoConnect et comprendre les raisons des départs,
puis exposer le modèle via une application Streamlit.

---

## 🎯 Objectifs

- Identifier les facteurs clés expliquant le churn
- Construire un modèle pour anticiper les clients à risque
- Aider les équipes Marketing & Customer Success à cibler leurs actions
- Fournir une application simple pour scorer un client

---

## 📈 Résultats clés

| Métrique (test, 16 609 clients) | Valeur |
|---------------------------------|--------|
| Modèle retenu | **XGBoost** |
| ROC-AUC | **0.79** |
| Recall (churners détectés) | **0.90** |
| F1-score | **0.78** |

**Facteur n°1 du churn : le contrat mensuel** — `IsMonthly` + `Contract`
concentrent **~92 %** du pouvoir prédictif. Un client mensuel a ~67 % de risque
de churn ; en contrat 2 ans, ~6 %.

➡️ Détails et plan d'action : [`reports/business_recommendations.md`](reports/business_recommendations.md)

---

## 🗂️ Structure du projet

```
telcoconnect-churn/
├── data/                 # Données brutes, nettoyées, préprocessées (X/y train/test)
├── notebooks/            # 01 audit → 06 entraînement des modèles
├── src/
│   ├── train_models.py   # Entraînement, évaluation, sauvegarde des modèles
│   └── inference.py      # Feature engineering + prédiction (production)
├── app/
│   └── app.py            # Application Streamlit de scoring
├── models/               # preprocessor.pkl, best_model.pkl
├── reports/              # Figures, comparatif modèles, recommandations business
├── requirements.txt
└── README.md
```

---

## 🔄 Pipeline du projet

| Étape | Notebook / Script | Statut |
|-------|-------------------|--------|
| 1. Audit des données | `notebooks/01_data_audit.ipynb` | ✅ |
| 2. Nettoyage | `notebooks/02_data_cleaning.ipynb` | ✅ |
| 3. Feature engineering | `notebooks/03_data_feature_engineering.ipynb` | ✅ |
| 4. Analyse exploratoire (EDA) + tests stats | `notebooks/04_exploratory_data_analysis.ipynb` | ✅ |
| 5. Préprocessing (scaling/encoding, split) | `notebooks/05_data_formating_for_ML_training.ipynb` | ✅ |
| 6. Entraînement & évaluation des modèles | `notebooks/06_Churn_prediction_ML_Model_Training.ipynb` | ✅ |
| 7. Application Streamlit | `app/app.py` | ✅ |
| 8. Recommandations business | `reports/business_recommendations.md` | ✅ |

---

## 🚀 Comment lancer

### 1. Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. (Ré)entraîner les modèles

```bash
python src/train_models.py
```
Produit : `models/best_model.pkl`, `reports/model_comparison.csv`,
`reports/metrics.json` et les figures dans `reports/figures/`.

### 3. Lancer l'application de scoring

```bash
streamlit run app/app.py
```
Saisir les caractéristiques d'un client → probabilité de churn + recommandation.
Le seuil de décision est ajustable dans la barre latérale.

### 4. (Optionnel) Explorer le notebook d'entraînement

```bash
jupyter notebook notebooks/06_Churn_prediction_ML_Model_Training.ipynb
```

---

## 🧠 Choix techniques

- **Données quasi équilibrées** (~47 % de churn) → pas de rééchantillonnage ;
  on privilégie **Recall** et **ROC-AUC** plutôt que l'Accuracy.
- **Cross-validation stratifiée (5 plis)** pour comparer les modèles de façon robuste.
- **RandomizedSearchCV** pour le tuning — qui n'a pas amélioré le modèle ici
  (signal concentré dans une seule variable) : le **XGBoost de base** est retenu.
- **`src/inference.py`** réplique exactement la logique de feature engineering des
  notebooks 03 & 05 → garantit la cohérence train / production.

---

**Auteur :** Igor Laminsi
