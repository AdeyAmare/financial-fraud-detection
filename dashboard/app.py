import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import shap
import os
import sys
from pathlib import Path

# -----------------------------
# PATH SETUP
# -----------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data.loader import FraudDataLoader, IPCountryLoader
from src.data.cleaner import TransactionDataCleaner
from src.data.merger import GeoDataMerger
from src.feature_engineering import FraudFeatureEngineer
from src.modeling import ModelingPipeline, PipelineConfig

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Fraud Risk Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Fraud Risk & Insights Dashboard")
st.write("Explore transactions, detect risky activity, and understand why some transactions may be risky. Everything is shown in plain language.")

# -----------------------------
# SIDEBAR UPLOAD
# -----------------------------
st.sidebar.header("Upload Your Data")
uploaded_txn = st.sidebar.file_uploader("Transaction Data (.csv)", type="csv")
uploaded_ip = st.sidebar.file_uploader("IP to Country Mapping (.csv)", type="csv")

# -----------------------------
# MAIN LOGIC
# -----------------------------
if uploaded_txn and uploaded_ip:
    try:
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as t1, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as t2:
            t1.write(uploaded_txn.getbuffer())
            t2.write(uploaded_ip.getbuffer())
            txn_path, ip_path = t1.name, t2.name

        with st.spinner("Preparing your data…"):
            df_raw = FraudDataLoader(txn_path).load()
            ip_map = IPCountryLoader(ip_path).load()
            df_clean = TransactionDataCleaner(df_raw).clean()
            df_geo = GeoDataMerger(df_clean, ip_map).merge_country()

            engineer = FraudFeatureEngineer(df_geo)
            engineer.add_time_features() \
                    .add_time_since_signup() \
                    .add_transaction_velocity()
            df = engineer.get_features()

        # -----------------------------
        # KEY METRICS
        # -----------------------------
        st.subheader("Summary Metrics")
        total_txns = len(df)
        fraud_txns = int(df["class"].sum())
        fraud_rate = fraud_txns / total_txns if total_txns else 0
        total_money = df["purchase_value"].sum()
        money_at_risk = df[df["class"] == 1]["purchase_value"].sum()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Transactions", f"{total_txns:,}")
        k2.metric("Potential Risky Transactions", f"{fraud_txns:,}", f"{fraud_rate:.1%}")
        k3.metric("Total Money Involved", f"${total_money/1e6:.2f}M")
        k4.metric("Money at Risk", f"${money_at_risk/1e3:.1f}K", delta_color="inverse")

        # -----------------------------
        # TABS
        # -----------------------------
        tab_eda, tab_model, tab_explain = st.tabs([
            "📊 Explore Data",
            "🤖 Fraud Detection",
            "🔍 Why Transactions Are Risky"
        ])

        # =============================
        # EDA
        # =============================
        with tab_eda:
            st.subheader("Explore Transactions in Simple Terms")
            st.plotly_chart(px.pie(
                df,
                names=df["class"].map({0: "Safe", 1: "Risky"}),
                title="Proportion of Safe vs Risky Transactions",
                hole=0.4
            ), use_container_width=True)

            st.plotly_chart(px.histogram(
                df,
                x="purchase_value",
                color=df["class"].map({0: "Safe", 1: "Risky"}),
                nbins=50,
                title="Transaction Amounts by Safety",
                labels={"color": "Transaction Type"}
            ), use_container_width=True)

            country_risk = df.groupby("country")["class"].mean().sort_values(ascending=False).reset_index()
            st.plotly_chart(px.bar(
                country_risk,
                x="country",
                y="class",
                title="Average Risk Level by Country",
                labels={"class": "Average Risk (0-1)"}
            ), use_container_width=True)

            st.plotly_chart(px.histogram(
                df,
                x="transactions_per_user",
                color=df["class"].map({0: "Safe", 1: "Risky"}),
                nbins=30,
                title="How Active Users Are",
                labels={"transactions_per_user": "Transactions per User"}
            ), use_container_width=True)

            st.plotly_chart(px.box(
                df,
                x="hour_of_day",
                y="class",
                title="Risky Transactions by Hour of Day",
                labels={"class": "Risk (0=Safe,1=Risky)", "hour_of_day": "Hour of Day"}
            ), use_container_width=True)

            st.plotly_chart(px.box(
                df,
                x="day_of_week",
                y="class",
                title="Risky Transactions by Day of Week",
                labels={"day_of_week": "Day of Week", "class": "Risk (0=Safe,1=Risky)"}
            ), use_container_width=True)

            st.plotly_chart(px.box(
                df,
                x="class",
                y="transactions_last_24h",
                title="Transactions in Last 24 Hours vs Risk",
                labels={"transactions_last_24h": "Transactions in Last 24h", "class": "Transaction Type"}
            ), use_container_width=True)

            st.plotly_chart(px.scatter(
                df,
                x="age",
                y="purchase_value",
                color=df["class"].map({0: "Safe", 1: "Risky"}),
                title="Purchase Amount vs Account Age",
                labels={"age": "User Account Age (days)", "purchase_value": "Transaction Amount"}
            ), use_container_width=True)

            browser_risk = df.groupby("browser")["class"].mean().sort_values(ascending=False).reset_index()
            st.plotly_chart(px.bar(
                browser_risk,
                x="browser",
                y="class",
                title="Average Risk by Browser",
                labels={"class": "Average Risk (0-1)"}
            ), use_container_width=True)

            source_risk = df.groupby("source")["class"].mean().sort_values(ascending=False).reset_index()
            st.plotly_chart(px.bar(
                source_risk,
                x="source",
                y="class",
                title="Average Risk by Source",
                labels={"class": "Average Risk (0-1)"}
            ), use_container_width=True)

        # =============================
        # MODEL PREDICTION
        # =============================
        with tab_model:
            st.subheader("How the System Detects Risky Transactions")

            config = PipelineConfig(
                numeric_features=[
                    "purchase_value", "age", "transactions_last_24h",
                    "transactions_per_user", "hour_of_day",
                    "day_of_week", "time_since_signup"
                ],
                categorical_features=["source", "browser", "country"]
            )

            pipeline = ModelingPipeline(df, config)
            pipeline.prepare_data()

            with st.spinner("Training model…"):
                lr_model = pipeline.tune_and_train_logistic_regression()
                rf_model = pipeline.train_random_forest()
                xgb_model = pipeline.train_xgboost()
                lgb_model = pipeline.train_lightgbm()
                best_model, best_name = (rf_model, "Random Forest")

            st.write(f"✅ Recommended model: **{best_name}**")
            st.write("This model predicts whether a transaction might be risky using simple patterns from transaction history, time, and user info.")

            # Performance comparison plot
            st.subheader("Compare Models in Simple Terms")
            df_preds = df.copy()
            X_all = pipeline.preprocessor.transform(df)
            df_preds["LR_prob"] = lr_model.predict_proba(X_all)[:, 1]
            df_preds["RF_prob"] = rf_model.predict_proba(X_all)[:, 1]
            df_preds["XGB_prob"] = xgb_model.predict_proba(X_all)[:, 1]
            df_preds["LGB_prob"] = lgb_model.predict_proba(X_all)[:, 1]
            df_preds["LR_pred"] = (df_preds["LR_prob"] > 0.5).astype(int)
            df_preds["RF_pred"] = (df_preds["RF_prob"] > 0.5).astype(int)
            df_preds["XGB_pred"] = (df_preds["XGB_prob"] > 0.5).astype(int)
            df_preds["LGB_pred"] = (df_preds["LGB_prob"] > 0.5).astype(int)

            model_perf = pd.DataFrame({
                "Model": ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"],
                "Risky Transactions Detected (%)": [
                    df_preds[df_preds["class"]==1]["LR_pred"].mean(),
                    df_preds[df_preds["class"]==1]["RF_pred"].mean(),
                    df_preds[df_preds["class"]==1]["XGB_pred"].mean(),
                    df_preds[df_preds["class"]==1]["LGB_pred"].mean()
                ]
            })

            st.plotly_chart(px.bar(
                model_perf,
                x="Model",
                y="Risky Transactions Detected (%)",
                text=model_perf["Risky Transactions Detected (%)"].apply(lambda x: f"{x:.1%}"),
                title="How Well Models Detect Risky Transactions",
                labels={"Risky Transactions Detected (%)":"Detected Risky %"}
            ), use_container_width=True)

            st.subheader("Check a Specific Transaction")
            idx = st.slider("Pick a transaction to check:", 0, len(df) - 1, 0)
            sample = df.iloc[[idx]]
            X_sample = pipeline.preprocessor.transform(sample)
            fraud_prob = best_model.predict_proba(X_sample)[0][1]

            st.metric("Estimated Risk", f"{fraud_prob:.1%}")
            if fraud_prob > 0.5:
                st.warning("🚨 This transaction is risky. Consider blocking it.")
            else:
                st.success("✅ This transaction looks safe.")

            st.write(sample.T)

        # =============================
        # EXPLAINABILITY
        # =============================
        with tab_explain:
            st.subheader("Why the Model Makes Predictions")

            X_processed = pipeline.preprocessor.transform(df)
            if hasattr(X_processed, "toarray"):
                X_processed = X_processed.toarray()
            feature_names = pipeline.preprocessor.get_feature_names_out()
            X_df = pd.DataFrame(X_processed, columns=feature_names)

            sample_size = min(200, X_df.shape[0])
            X_shap = X_df.sample(sample_size, random_state=42)

            explainer = shap.TreeExplainer(best_model)
            shap_values_raw = explainer.shap_values(X_shap)

            if isinstance(shap_values_raw, list):
                shap_values = shap_values_raw[1]
            else:
                shap_values = shap_values_raw
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]

            # -----------------------------
            # Interactive SHAP Beeswarm with Plotly
            # -----------------------------
            st.subheader("Overall Feature Impact (Interactive)")
            shap_df = pd.DataFrame(shap_values, columns=feature_names)
            shap_df["sample_id"] = np.arange(shap_df.shape[0])
            shap_long = shap_df.melt(id_vars="sample_id", var_name="feature", value_name="impact")

            fig = px.scatter(
                shap_long, x="impact", y="feature", color="impact", 
                hover_data=["sample_id"], title="SHAP Feature Impact (Beeswarm Style)",
                color_continuous_scale="RdBu", width=900, height=600
            )
            st.plotly_chart(fig, use_container_width=True)

            # -----------------------------
            # Local Explanation
            # -----------------------------
            st.subheader("Why This Transaction is Risky")
            idx_local = st.slider("Pick a transaction for explanation:", 0, len(df)-1, idx)
            X_single = X_df.iloc[[idx_local]]
            single_shap_raw = explainer.shap_values(X_single)

            if isinstance(single_shap_raw, list):
                single_shap = single_shap_raw[1]
            else:
                single_shap = single_shap_raw
            if single_shap.ndim == 3:
                single_shap = single_shap[:, :, 1]
            single_shap_flat = single_shap.flatten()

            single_df = pd.DataFrame({
                "feature": feature_names,
                "impact": single_shap_flat
            })
            single_df["abs_impact"] = single_df["impact"].abs()
            single_df = single_df.sort_values("abs_impact", ascending=False).head(10)

            st.plotly_chart(px.bar(
                single_df, 
                x="impact", 
                y="feature", 
                orientation="h",
                title=f"Top Factors for Transaction #{idx_local}",
                color="impact",
                color_continuous_scale="RdBu"
            ), use_container_width=True)

    except Exception as e:
        st.error(f"Something went wrong: {e}")

    finally:
        if "txn_path" in locals() and os.path.exists(txn_path):
            os.remove(txn_path)
        if "ip_path" in locals() and os.path.exists(ip_path):
            os.remove(ip_path)

else:
    st.info("Please upload both transaction and IP data to start exploring.")
