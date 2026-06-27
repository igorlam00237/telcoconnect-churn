# TelcoConnect — Recommandations business (Prédiction du churn)

**Auteur :** Igor Laminsi · **Date :** juin 2026
**Modèle en production :** XGBoost · ROC-AUC ≈ 0.79 · Recall ≈ 0.90

---

## 1. Synthèse exécutive

Un modèle de Machine Learning a été entraîné sur **66 433 clients** pour prédire
le risque de résiliation (churn). Le modèle identifie **~90 % des clients qui
vont effectivement partir** (recall 0.90), ce qui en fait un outil de ciblage
fiable pour les campagnes de rétention.

**Le message principal est sans ambiguïté : le type de contrat est, de très loin,
le premier facteur de churn.** À lui seul, le fait d'être en contrat *mensuel*
explique l'essentiel des départs.

---

## 2. Top facteurs de churn (feature importance)

| Rang | Facteur | Importance | Lecture |
|------|---------|-----------|---------|
| 1 | **IsMonthly** (contrat mensuel) | 0.49 | Un client mensuel est massivement plus à risque |
| 2 | **Contract** (durée d'engagement) | 0.43 | Confirme le rôle central de l'engagement |
| 3+ | Services, charges, ancienneté… | < 0.01 chacun | Effets marginaux |

➡️ **Les deux premiers facteurs concentrent ~92 % du pouvoir prédictif.**
Tout le reste (services souscrits, montant facturé, profil démographique) joue
un rôle secondaire.

**Illustration concrète** (sorties du modèle) :
- Client *mensuel*, faible ancienneté, fibre : **~67 % de probabilité de churn**
- Même client en contrat *2 ans*, ancienneté 60 mois : **~6 % de probabilité de churn**

---

## 3. Recommandations actionnables

### 🎯 Action n°1 — Convertir les contrats mensuels vers de l'engagement
Le levier le plus rentable. Cibler en priorité les clients **mensuels à faible
ancienneté** avec :
- Une remise conditionnée au passage en contrat 1 an / 2 ans.
- Des avantages exclusifs « engagement » (mois offert, options incluses).
- Un parcours de réengagement automatisé dès les premiers mois.

### 🎯 Action n°2 — Programme d'onboarding sur les 12 premiers mois
L'ancienneté faible amplifie le risque. Renforcer l'accompagnement la première
année (check-ins, support proactif) pour franchir le cap critique.

### 🎯 Action n°3 — Scorer et prioriser via l'app
Utiliser l'application Streamlit (`app/app.py`) pour scorer les clients et
alimenter les équipes Marketing / Customer Success avec une **liste priorisée**
des clients à risque.

---

## 4. Calibration du seuil selon le coût métier

Le modèle renvoie une **probabilité**. Le seuil de décision se règle selon
l'arbitrage coût d'une action de rétention vs coût d'un client perdu :

| Seuil | Recall (churners détectés) | Precision | Usage recommandé |
|-------|---------------------------|-----------|------------------|
| 0.30 | **90 %** | 70 % | Campagne large, on accepte des fausses alertes |
| 0.40–0.45 | ~90 % | ~70 % | **Équilibre recommandé (F1 max ≈ 0.785)** |
| 0.50 | 90 % | 70 % | Défaut |
| 0.65 | 80 % | 70 % | Budget rétention limité, on cible le plus sûr |

> Un *faux négatif* (churner manqué = client perdu) coûte généralement plus cher
> qu'un *faux positif* (offre de rétention inutile). → privilégier un **seuil bas
> (0.30–0.40)** pour maximiser la détection.

---

## 5. Performance du modèle (jeu de test, 16 609 clients)

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.770 | 0.698 | 0.898 | 0.785 | 0.791 |
| Random Forest | 0.760 | 0.696 | 0.866 | 0.772 | 0.790 |
| **XGBoost (retenu)** | **0.769** | **0.698** | **0.896** | **0.785** | **0.793** |

**Note méthodologique :** les trois familles sont quasi équivalentes et le tuning
n'a pas amélioré le modèle. C'est cohérent avec une information prédictive
concentrée dans une seule variable (le contrat). Un modèle simple aurait presque
suffi ; XGBoost est retenu pour sa légère avance et sa robustesse.

---

## 6. Limites & pistes d'amélioration

- **Plafond de performance (~0.79 AUC)** : les features actuelles capturent surtout
  le contrat. Pour progresser, collecter des signaux comportementaux
  (réclamations, usage réel, interactions support, incidents réseau).
- `IsMonthly` et `Contract` sont **redondants** — on pourrait n'en garder qu'un.
- Surveiller le **drift** du modèle dans le temps et le ré-entraîner périodiquement.

---

*Artefacts : `models/best_model.pkl`, figures dans `reports/figures/`,
notebook `notebooks/06_Churn_prediction_ML_Model_Training.ipynb`, app `app/app.py`.*
