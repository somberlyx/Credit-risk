import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="💳",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3rem;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        margin-top: 2rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
        border: none;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
        text-align: center;
    }
    .good-risk {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .bad-risk {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .info-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and encoders
@st.cache_resource
def load_models():
    model = joblib.load("best_xgb_model.pkl")
    encoder = {
        col: joblib.load(f"{col}_le.pkl") 
        for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
    }
    return model, encoder

model, encoder = load_models()

# Header
st.title("💳 Credit Risk Assessment Tool")
st.markdown('<p class="subtitle">AI-powered credit risk evaluation system</p>', unsafe_allow_html=True)

# Info card
st.markdown("""
    <div class="info-card" style="background-color:black;">
        <h3>📊 How it works</h3>
        <p>This tool uses machine learning to assess credit risk based on applicant information. 
        Fill in the details below to get an instant risk assessment.</p>
    </div>
    """, unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Information")
    age = st.slider("Age", min_value=18, max_value=100, value=30, help="Applicant's age in years")
    sex = st.selectbox("Gender", encoder["Sex"].classes_, help="Applicant's gender")
    job = st.select_slider(
        "Job Level", 
        options=[0, 1, 2, 3], 
        value=1,
        help="Job classification: 0=Unskilled, 1=Skilled, 2=Highly Skilled, 3=Management"
    )
    housing = st.selectbox("Housing Status", encoder["Housing"].classes_, help="Current housing situation")

with col2:
    st.subheader("💰 Financial Information")
    credit_amount = st.number_input(
        "Credit Amount ($)", 
        min_value=0, 
        value=5000, 
        step=100,
        help="Amount of credit requested"
    )
    duration = st.slider(
        "Duration (months)", 
        min_value=1, 
        max_value=72, 
        value=12,
        help="Loan duration in months"
    )
    saving_accounts = st.selectbox(
        "Savings Account Status", 
        encoder["Saving accounts"].classes_,
        help="Current savings account level"
    )
    checking_account = st.selectbox(
        "Checking Account Status", 
        encoder["Checking account"].classes_,
        help="Current checking account level"
    )

# Create input DataFrame
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

# Predict button
if st.button("🔍 Assess Credit Risk"):
    with st.spinner("Analyzing credit risk..."):
        prediction = model.predict(input_df)[0]
        
        # Try to get probability if available
        try:
            probability = model.predict_proba(input_df)[0]
            prob_good = probability[1] * 100
            prob_bad = probability[0] * 100
        except:
            prob_good = None
            prob_bad = None
        
        # Display results
        if prediction == 1:
            st.markdown("""
                <div class="result-box good-risk">
                    <h2>✅ LOW RISK - APPROVED</h2>
                    <p style="font-size: 1.2rem; margin-top: 1rem;">
                        This applicant shows <strong>good credit characteristics</strong> and is likely to repay the loan.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            if prob_good:
                st.metric(
                    label="Confidence Level", 
                    value=f"{prob_good:.1f}%",
                    delta="Good Risk"
                )
        else:
            st.markdown("""
                <div class="result-box bad-risk">
                    <h2>⚠️ HIGH RISK - REVIEW REQUIRED</h2>
                    <p style="font-size: 1.2rem; margin-top: 1rem;">
                        This applicant shows <strong>elevated risk characteristics</strong>. Further review recommended.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            if prob_bad:
                st.metric(
                    label="Confidence Level", 
                    value=f"{prob_bad:.1f}%",
                    delta="High Risk",
                    delta_color="inverse"
                )
        
        # Show input summary
        with st.expander("📋 View Application Summary"):
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.write("**Personal Details:**")
                st.write(f"- Age: {age} years")
                st.write(f"- Gender: {sex}")
                st.write(f"- Job Level: {job}")
                st.write(f"- Housing: {housing}")
            
            with summary_col2:
                st.write("**Financial Details:**")
                st.write(f"- Credit Amount: ${credit_amount:,}")
                st.write(f"- Duration: {duration} months")
                st.write(f"- Savings: {saving_accounts}")
                st.write(f"- Checking: {checking_account}")

# Sidebar with additional info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This credit risk assessment tool uses an XGBoost machine learning model 
    to predict the likelihood of loan repayment.
    """)
    
    st.header("📈 Risk Levels")
    st.success("**Good (1)**: Lower risk, likely to repay")
    st.error("**Bad (0)**: Higher risk, may default")
    
    with st.expander("🔧 Model Information"):
        st.write("**Features used:**")
        st.write("- Age, Gender, Job Level")
        st.write("- Housing Status")
        st.write("- Account Balances")
        st.write("- Credit Amount & Duration")
    
    with st.expander("🐛 Debug Info"):
        st.write("**Encoder Classes:**")
        for col, enc in encoder.items():
            st.write(f"**{col}:** {', '.join(enc.classes_)}")