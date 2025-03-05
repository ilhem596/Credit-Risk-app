import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le modèle et le scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

# Titre de l'application
st.title("Prédiction du Risque de Crédit")
st.markdown("*Cette application prédit si un client est risqué ou non en fonction de ses caractéristiques.*")

# Interface utilisateur : saisie des caractéristiques du client
person_age = st.number_input("Âge de la personne:", value=30)
person_income = st.number_input("Revenu annuel:", value=50000)
person_home_ownership = st.selectbox("Type de propriété:", ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
person_emp_length = st.number_input("Durée d'emploi (années):", value=5)
loan_intent = st.selectbox("Intention du prêt:", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
loan_grade = st.selectbox("Grade du prêt:", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
loan_amnt = st.number_input("Montant du prêt:", value=10000)
loan_int_rate = st.number_input("Taux d'intérêt:", value=10.0)
loan_percent_income = st.number_input("Pourcentage du revenu alloué au prêt:", value=0.2)
cb_person_default_on_file = st.selectbox("Historique de défaut de paiement:", ['Y', 'N'])
cb_person_cred_hist_length = st.number_input("Longueur historique crédit (années):", value=5)

# Encodage des variables catégorielles
mapping_home_ownership = {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'OTHER': 3}
mapping_loan_intent = {'EDUCATION': 0, 'MEDICAL': 1, 'VENTURE': 2, 'PERSONAL': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5}
mapping_loan_grade = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
mapping_default = {'N': 0, 'Y': 1}

user_data = {
    "person_age": person_age,
    "person_income": person_income,
    "person_home_ownership": mapping_home_ownership[person_home_ownership],
    "person_emp_length": person_emp_length,
    "loan_intent": mapping_loan_intent[loan_intent],
    "loan_grade": mapping_loan_grade[loan_grade],
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_default_on_file": mapping_default[cb_person_default_on_file],
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
}

user_df = pd.DataFrame([user_data])

# Normalisation des données
user_df_scaled = scaler.transform(user_df)

# Bouton de prédiction
if st.button("Prédire le risque de crédit"):
    prediction = model.predict(user_df_scaled)
    proba = model.predict_proba(user_df_scaled)[0][1] * 100
    
    if prediction[0] == 1:
        st.error(f"Le client est considéré comme risqué avec une probabilité de {proba:.2f}%.")
    else:
        st.success(f"Le client est considéré comme non risqué avec une probabilité de {100 - proba:.2f}%.")
