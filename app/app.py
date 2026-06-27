"""
TelcoConnect — Application de prédiction du churn
=================================================

Lancer :  ./venv/bin/streamlit run app/app.py

L'app saisit les caractéristiques d'un client, recalcule les features dérivées
(via src/inference.py — même logique que les notebooks 03 & 05), applique le
preprocessor sauvegardé et le meilleur modèle (XGBoost), puis affiche la
probabilité de churn et une recommandation d'action.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Rendre src/ importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference import CHOICES, SERVICE_COLS, build_raw_row  # noqa: E402

MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

st.set_page_config(page_title="TelcoConnect — Churn", page_icon="📊", layout="wide")


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(MODELS / "preprocessor.pkl")
    model = joblib.load(MODELS / "best_model.pkl")
    meta = {}
    meta_path = MODELS / "best_model_meta.pkl"
    if meta_path.exists():
        meta = joblib.load(meta_path)
    return preprocessor, model, meta


@st.cache_data
def load_feature_importance():
    fi_path = REPORTS / "feature_importance.csv"
    if fi_path.exists():
        return pd.read_csv(fi_path)
    return None


preprocessor, model, meta = load_artifacts()

# --------------------------------------------------------------------------- #
# En-tête
# --------------------------------------------------------------------------- #
st.title("📊 TelcoConnect — Prédiction du churn client")
st.caption(
    f"Modèle en production : **{meta.get('best_label', 'XGBoost')}** · "
    "ROC-AUC ≈ 0.79 · Recall ≈ 0.90"
)

# --------------------------------------------------------------------------- #
# Sidebar : seuil de décision
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Paramètres")
    threshold = st.slider(
        "Seuil de décision (churn si proba ≥ seuil)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help=(
            "Baisser le seuil = détecter plus de churners (recall ↑) mais plus "
            "de fausses alertes. À calibrer selon le coût d'une rétention vs "
            "le coût d'un client perdu."
        ),
    )
    st.markdown("---")
    fi = load_feature_importance()
    if fi is not None:
        st.subheader("🔑 Drivers du churn")
        st.caption("Importance des features (modèle)")
        st.bar_chart(fi.set_index("feature")["importance"].head(8))

# --------------------------------------------------------------------------- #
# Formulaire de saisie
# --------------------------------------------------------------------------- #
st.subheader("👤 Caractéristiques du client")

with st.form("client_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Profil**")
        gender = st.selectbox("Genre", CHOICES["gender"])
        senior = st.selectbox("Senior (65+)", CHOICES["yes_no"])
        partner = st.selectbox("En couple (Partner)", CHOICES["yes_no"])
        dependents = st.selectbox("A des personnes à charge", CHOICES["yes_no"])
        tenure = st.slider("Ancienneté (mois)", 1, 72, 12)

    with c2:
        st.markdown("**Contrat & facturation**")
        contract = st.selectbox("Type de contrat", CHOICES["Contract"])
        payment = st.selectbox("Moyen de paiement", CHOICES["PaymentMethod"])
        paperless = st.selectbox("Facture dématérialisée", CHOICES["yes_no"])
        monthly = st.number_input(
            "Charges mensuelles ($)", min_value=18.0, max_value=120.0, value=70.0, step=1.0
        )
        total = st.number_input(
            "Charges totales ($)", min_value=1.0, max_value=9000.0, value=840.0, step=10.0
        )

    with c3:
        st.markdown("**Services**")
        phone = st.selectbox("Téléphonie", CHOICES["yes_no"], index=1)
        multilines = st.selectbox("Lignes multiples", CHOICES["MultipleLines"])
        internet = st.selectbox("Internet", CHOICES["InternetService"])
        services = {}
        for svc in SERVICE_COLS:
            services[svc] = st.selectbox(svc, CHOICES["yes_no"], key=svc)

    submitted = st.form_submit_button("🔮 Prédire le churn", use_container_width=True)

# --------------------------------------------------------------------------- #
# Prédiction
# --------------------------------------------------------------------------- #
if submitted:
    inp = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multilines,
        "InternetService": internet,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        **services,
    }

    raw = build_raw_row(inp)
    X = preprocessor.transform(raw)
    proba = float(model.predict_proba(X)[0, 1])
    is_churn = proba >= threshold

    st.markdown("---")
    r1, r2 = st.columns([1, 2])

    with r1:
        st.metric("Probabilité de churn", f"{proba:.0%}")
        if is_churn:
            st.error("⚠️ Client À RISQUE de résiliation")
        else:
            st.success("✅ Client probablement fidèle")

    with r2:
        st.progress(min(proba, 1.0))
        if proba >= 0.7:
            level, advice = "🔴 Risque élevé", (
                "Action immédiate : appel de rétention, offre de migration vers "
                "un contrat annuel/2 ans, remise ciblée."
            )
        elif proba >= 0.4:
            level, advice = "🟠 Risque modéré", (
                "Surveiller : proposer des services additionnels (sécurité, "
                "support) et inciter à un engagement plus long."
            )
        else:
            level, advice = "🟢 Risque faible", (
                "Fidélisation standard : programme de récompenses, upsell doux."
            )
        st.markdown(f"### {level}")
        st.info(advice)

    with st.expander("🔍 Détail des features recalculées (debug)"):
        st.dataframe(raw.T.rename(columns={0: "valeur"}))

st.markdown("---")
st.caption(
    "TelcoConnect Churn · Modèle entraîné sur 49 824 clients · "
    "⚠️ Le type de contrat (mensuel) est le facteur dominant du churn."
)
