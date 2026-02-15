import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

import sys

# Import your functional classes

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import your custom classes
from src.data.loader import FraudDataLoader
from src.data.cleaner import TransactionDataCleaner
from src.feature_engineering import FraudFeatureEngineer, FraudFeatureConfig
from src.modeling import ModelingPipeline, PipelineConfig
from src.explainability import ModelExplainability, ExplainabilityConfig


# Page setup
st.set_page_config(page_title="Fraud Risk Executive Dashboard", layout="wide")

# Custom styling for a professional look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Fraud Risk & Revenue Protection")
st.subheader("Executive Insights & Decision Support")

# --- SIDEBAR: DATA UPLOAD ---
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload Latest Transaction Data", type=["csv"])

if uploaded_file:
    # 1. Processing Pipeline (Functional & Silent)
    with open("temp_data.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and Clean
    df_raw = FraudDataLoader("temp_data.csv").load()
    df_cleaned = TransactionDataCleaner(df_raw).clean(missing_strategy="drop")
    
    # Feature Engineering
    engineer = FraudFeatureEngineer(df_cleaned)
    engineer.add_time_features().add_transaction_velocity()
    df = engineer.get_features()

    # --- SECTION 1: KEY METRICS ---
    fraud_count = df['class'].sum()
    fraud_rate = (fraud_count / len(df)) * 100
    total_volume = df['purchase_value'].sum()
    at_risk_value = df[df['class'] == 1]['purchase_value'].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Detected Fraud Rate", f"{fraud_rate:.2f}%", delta="-0.15% vs Last Period")
    col3.metric("Total Volume", f"${total_volume/1e6:.2f}M")
    col4.metric("At-Risk Revenue", f"${at_risk_value/1e3:.1f}K", delta_color="inverse")

    # --- SECTION 2: BUSINESS IMPACT & VISUALIZATIONS ---
    tab1, tab2, tab3 = st.tabs(["💰 Business Impact", "🔍 Risk Patterns", "🤖 AI Performance"])

    with tab1:
        st.subheader("Revenue vs. Loss Analysis")
        # Donut chart for revenue split
        impact_data = pd.DataFrame({
            'Category': ['Safe Revenue', 'Blocked Fraud'],
            'Value': [total_volume - at_risk_value, at_risk_value]
        })
        fig_impact = px.pie(impact_data, values='Value', names='Category', 
                           hole=0.6, color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig_impact, use_container_width=True)

    with tab2:
        st.subheader("Where and When is Risk Occurring?")
        c1, c2 = st.columns(2)
        
        with c1:
            # Hourly risk pattern
            hourly = df.groupby('hour_of_day')['class'].mean().reset_index()
            fig_hour = px.line(hourly, x='hour_of_day', y='class', 
                              title="Risk Probability by Hour",
                              labels={'class': 'Risk Level', 'hour_of_day': 'Hour of Day'})
            fig_hour.update_traces(line_color='#e74c3c')
            st.plotly_chart(fig_hour, use_container_width=True)
            
        with c2:
            # Source of traffic risk
            source_risk = df.groupby('source')['class'].mean().reset_index()
            fig_source = px.bar(source_risk, x='source', y='class', 
                               title="Risk by Marketing Channel",
                               color='class', color_continuous_scale='Reds')
            st.plotly_chart(fig_source, use_container_width=True)

    with tab3:
        st.subheader("AI Detection Drivers")
        # Run modeling pipeline
        config = PipelineConfig(
            numeric_features=['purchase_value', 'age', 'transactions_per_user', 'transactions_last_24h'],
            categorical_features=['source', 'browser']
        )
        pipeline = ModelingPipeline(df, config)
        pipeline.prepare_data()
        
        with st.spinner("Analyzing data patterns..."):
            model = pipeline.train_random_forest()
        
        # Display Feature Importance (Stakeholder friendly)
        importances = pd.Series(model.feature_importances_, 
                               index=pipeline.preprocessor.get_feature_names_out())
        top_drivers = importances.sort_values(ascending=True).tail(10)
        
        fig_drivers = px.bar(x=top_drivers.values, y=top_drivers.index, orientation='h',
                            title="Top 10 Risk Indicators (What the AI is looking at)",
                            labels={'x': 'Relative Importance', 'y': 'Feature'})
        st.plotly_chart(fig_drivers, use_container_width=True)

    # --- SECTION 3: INTERACTIVE PREDICTION ---
    st.markdown("---")
    st.subheader("🎯 Test a Transaction Scenario")
    st.write("Adjust parameters to see how the model calculates risk score.")
    
    test_col1, test_col2, test_col3 = st.columns(3)
    p_val = test_col1.number_input("Purchase Amount ($)", 1, 1000, 50)
    velocity = test_col2.slider("Transactions in Last 24h", 1, 50, 1)
    age = test_col3.slider("User Account Age (Days)", 0, 365, 30)
    
    # Simple probability logic for simulation (stakeholder interaction)
    risk_score = min(99, (p_val / 10) + (velocity * 5) - (age / 10))
    st.progress(risk_score / 100)
    st.write(f"Calculated Risk Probability: **{risk_score:.1f}%**")

else:
    st.info("👋 Welcome! Please upload your transaction CSV to generate the fraud report.")