import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# -------------------- Page Config --------------------
st.set_page_config(page_title="Credit Risk Dashboard & Prediction", layout="wide")

st.title("💳 Credit Risk Interactive Dashboard & Prediction App")

# -------------------- Load Saved Model Stuff --------------------
@st.cache_resource
def load_model_objects():
    model = joblib.load("best_credit_risk_model.pkl")
    encoder = joblib.load("label_encoders.pkl")
    scaler = joblib.load("minmax_scaler.pkl")
    return model, encoder, scaler

model, encoder, scaler = load_model_objects()

# -------------------- Upload Dataset for Dashboard --------------------
st.sidebar.header("📂 Upload Dataset for Dashboard")
dataset_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

categorical_cols = ['employment_type','education_level','region','device_type']
numerical_cols = [
    'age','monthly_income','debt_ratio','credit_utilization',
    'transaction_count_30d','avg_transaction_amount',
    'last_payment_delay_days','internal_score_v2'
]

if dataset_file:
    df = pd.read_csv(dataset_file)
    st.success("Dataset Loaded Successfully ✔️")

    # ---------------- Sidebar Filters ----------------
    st.sidebar.header("🔎 Filters")
    region_filter = st.sidebar.multiselect("Select Region", options=df["region"].unique(), default=df["region"].unique())
    emp_filter = st.sidebar.multiselect("Employment Type", options=df["employment_type"].unique(), default=df["employment_type"].unique())
    df = df[(df["region"].isin(region_filter)) & (df["employment_type"].isin(emp_filter))]

    # ---------------- Tabs ----------------
    tabs = st.tabs(["📊 Overview",
                    "👤 Customer Profile Insights",
                    "💳 Credit Behavior",
                    "⚖️ Risk & Imbalance",
                    "🤖 Model Prediction"])

    # ---------------- OVERVIEW ----------------
    with tabs[0]:
        st.subheader("📊 Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", len(df))
        c2.metric("Avg Monthly Income", round(df["monthly_income"].mean(), 2))
        c3.metric("Avg Internal Score", round(df["internal_score_v2"].mean(), 2))

        st.dataframe(df.head())

        fig = px.pie(df, names="target", title="Target Class Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- CUSTOMER PROFILE ----------------
    with tabs[1]:
        st.subheader("👤 Customer Profile Insights")

        fig = px.histogram(df, x="age", color="target", nbins=40, title="Age Distribution by Risk")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.box(df, x="education_level", y="monthly_income",
                     color="target", title="Income by Education Level")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- CREDIT BEHAVIOR ----------------
    with tabs[2]:
        st.subheader("💳 Credit Behavior Analysis")

        fig = px.scatter(df,
                         x="debt_ratio",
                         y="credit_utilization",
                         color="target",
                         title="Debt Ratio vs Credit Utilization")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(df,
                         x="transaction_count_30d",
                         y="avg_transaction_amount",
                         color="target",
                         title="Transaction Count vs Avg Amount")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- RISK & IMBALANCE ----------------
    with tabs[3]:
        st.subheader("⚖️ Risk & Imbalance View")

        fig = px.box(df, x="target", y="internal_score_v2",
                     title="Internal Score by Risk Class")
        st.plotly_chart(fig, use_container_width=True)

        # Fix bar chart for employment_type
        emp_count = df['employment_type'].value_counts().reset_index()
        emp_count.columns = ['employment_type','count']
        fig = px.bar(emp_count, x='employment_type', y='count', title="Employment Type Distribution")
        st.plotly_chart(fig, use_container_width=True)

        # Region distribution
        reg_count = df['region'].value_counts().reset_index()
        reg_count.columns = ['region','count']
        fig = px.bar(reg_count, x='region', y='count', title="Region Distribution")
        st.plotly_chart(fig, use_container_width=True)

        # Education level distribution
        edu_count = df['education_level'].value_counts().reset_index()
        edu_count.columns = ['education_level','count']
        fig = px.bar(edu_count, x='education_level', y='count', title="Education Level Distribution")
        st.plotly_chart(fig, use_container_width=True)

        # Device type distribution
        dev_count = df['device_type'].value_counts().reset_index()
        dev_count.columns = ['device_type','count']
        fig = px.bar(dev_count, x='device_type', y='count', title="Device Type Distribution")
        st.plotly_chart(fig, use_container_width=True)



# ---------------- MODEL PREDICTION ----------------
with st.container():
    st.header("🤖 Credit Risk Prediction (Using Saved Model)")
    st.info("This prediction uses pre-trained model.pkl, encoder.pkl and scaler.pkl")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18.0, 90.0, 30.0)
        income = st.number_input("Monthly Income", 0.0, 500000.0, 30000.0)
        debt = st.number_input("Debt Ratio", 0.0, 2.0, 0.3)
        util = st.number_input("Credit Utilization", 0.0, 2.0, 0.5)
        tx = st.number_input("Transaction Count (30d)", 0, 200, 40)

    with col2:
        avg_tx = st.number_input("Avg Transaction Amount", 0.0, 5000.0, 200.0)
        delay = st.number_input("Last Payment Delay (days)", 0.0, 60.0, 2.0)
        score = st.number_input("Internal Score v2", 300.0, 900.0, 600.0)

        emp = st.selectbox("Employment Type", encoder['employment_type'].classes_)
        edu = st.selectbox("Education Level", encoder['education_level'].classes_)
        region = st.selectbox("Region", encoder['region'].classes_)
        device = st.selectbox("Device Type", encoder['device_type'].classes_)

    if st.button("🔍 Predict Credit Risk"):
        # Define column order exactly as used during training
        feature_order = [
            'age','monthly_income','debt_ratio','credit_utilization',
            'transaction_count_30d','avg_transaction_amount',
            'last_payment_delay_days','internal_score_v2',
            'employment_type','education_level','region','device_type'
        ]

        # Create sample dataframe
        sample = pd.DataFrame([[
            age, income, debt, util, tx, avg_tx,
            delay, score, emp, edu, region, device
        ]], columns=feature_order)

        # Encode categorical columns
        categorical_cols = ['employment_type','education_level','region','device_type']
        for col in categorical_cols:
            sample[col] = encoder[col].transform(sample[col])

        # Ensure correct column order
        sample = sample[feature_order]

        # Convert to numpy array to avoid feature name mismatch
        sample_scaled = scaler.transform(sample.values)

        # Predict
        pred_class = model.predict(sample_scaled)[0]
        pred_prob = model.predict_proba(sample_scaled)[0][1]

        # Display result
        if pred_class == 1:
            st.error(f"⚠️ High Credit Risk | Probability = {round(pred_prob,3)}")
        else:
            st.success(f"✅ Low Credit Risk | Probability = {round(pred_prob,3)}")
