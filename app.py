# 1 = Good (Lower Risk), 0 = Bad (Higher Risk)
import streamlit as st
import joblib
import pandas as pd

model = joblib.load("best_xgb_model.pkl")
encoder = {col: joblib.load(f"{col}_encoder.pkl") for col in ["Sex", "Housing", "Saving accounts", "Checking account"]}

# Display what values the encoders expect (for debugging)
st.sidebar.write("### Encoder Classes (for debugging)")
for col, enc in encoder.items():
    st.sidebar.write(f"{col}: {enc.classes_}")

st.title("Credit Risk Prediction")
st.write("Enter the details to predict credit risk (1 = Good, 0 = Bad)")

age = st.number_input("Age", min_value=18, max_value=100, value=30) 
sex = st.selectbox("Sex", encoder["Sex"].classes_)
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", encoder["Housing"].classes_)
saving_accounts = st.selectbox("Saving accounts", encoder["Saving accounts"].classes_)
checking_account = st.selectbox("Checking account", encoder["Checking account"].classes_)
credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.number_input("Duration (months)", min_value=1, value=12)

# Create DataFrame with columns in the EXACT order the model expects
input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoder["Sex"].transform([sex])[0]],
    "Job": [job],
    "Credit amount": [credit_amount],
    "Duration": [duration],
    "Saving accounts": [encoder["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoder["Checking account"].transform([checking_account])[0]],
    "Housing": [encoder["Housing"].transform([housing])[0]]
})

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("✅ The credit risk is GOOD (Lower Risk).")
    else:
        st.error("⚠️ The credit risk is BAD (Higher Risk).")