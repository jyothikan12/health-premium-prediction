import streamlit as st
import pandas as pd
from prediction_helper import predict

st.set_page_config(page_title="Health Insurance Cost Predictor", layout="wide")
st.title("🏥 Health Insurance Cost Predictor")
st.write("Enter your details below to see the input data dictionary printed in the terminal.")

# 3-column layout
col1, col2, col3 = st.columns(3)

# --- Column 1 ---
with col1:
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    number_of_dependants = st.number_input("Number of Dependants", min_value=0, max_value=10, value=0)
    income_in_lakhs = st.number_input("Income (in Lakhs)", min_value=0.0, step=0.5)
    genetical_risk = st.number_input("Genetical Risk", min_value=0, max_value=10, value=2, step=1)

# --- Column 2 ---
with col2:
    insurance_plan = st.selectbox("Insurance Plan", ['Silver', 'Bronze', 'Gold'])
    employment_status = st.selectbox("Employment Status", ['Self-Employed', 'Freelancer', 'Salaried'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    marital_status = st.selectbox("Marital Status", ['Unmarried', 'Married'])

# --- Column 3 ---
with col3:
    bmi_category = st.selectbox("BMI Category", ['Overweight', 'Underweight', 'Normal', 'Obesity'])
    smoking_status = st.selectbox("Smoking Status", [
        'Regular', 'No Smoking', 'Occasional', 'Smoking=0', 'Does Not Smoke', 'Not Smoking'
    ])
    region = st.selectbox("Region", ['Northeast', 'Northwest', 'Southeast', 'Southwest'])
    medical_history = st.selectbox("Medical History", [
        'High blood pressure', 'No Disease', 'Diabetes & High blood pressure',
        'Diabetes & Heart disease', 'Diabetes', 'Diabetes & Thyroid',
        'Heart disease', 'Thyroid', 'High blood pressure & Heart disease'
    ])

st.markdown("---")

# Predict button
input_data = {
    'age': age,
    'number_of_dependants': number_of_dependants,
    'income_in_lakhs': income_in_lakhs,
    'genetical_risk': genetical_risk,
    'insurance_plan': insurance_plan,
    'employment_status': employment_status,
    'gender': gender,
    'marital_status': marital_status,
    'bmi_category': bmi_category,
    'smoking_status': smoking_status,
    'region': region,
    'medical_history': medical_history
    }

if st.button('Predict'):
    prediction = predict(input_data)
    st.success(f"Predicted Premium: {prediction}")
