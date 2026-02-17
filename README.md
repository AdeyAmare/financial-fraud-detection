
# Financial Fraud Detection & Risk Intelligence Platform

This project implements a production-ready fraud detection system for e-commerce and credit card transactions. It uses behavioral feature engineering, machine learning, explainable AI, and a stakeholder-focused dashboard to detect fraud accurately while minimizing false alerts to legitimate customers.

---

## Business Problem

Financial institutions and e-commerce platforms face significant losses due to fraud every year. The main challenge is to detect fraudulent transactions without creating unnecessary friction for legitimate users. Missing fraud causes direct financial loss, whereas over-flagging legitimate transactions damages customer trust and increases operational costs.

This project addresses the need for a system that balances these competing objectives, provides actionable insights for risk teams, and enables rapid operational response. The solution also aims to make predictions interpretable, ensuring transparency and regulatory compliance.

---

## Solution Overview

The approach begins with ingestion and cleaning of transaction datasets, followed by feature engineering to capture user behavior, transaction timing, geographic signals, and velocity patterns. Imbalanced classes are addressed using SMOTE applied only to the training set to prevent data leakage.

Models are trained and evaluated using Logistic Regression as a baseline, followed by ensemble methods such as Random Forest, XGBoost, and LightGBM. Model selection is driven by metrics aligned with business priorities, such as precision, recall, F1-score, and area under the precision-recall curve (AUC-PR), rather than raw accuracy alone.

A Streamlit dashboard was developed for non-technical stakeholders to upload data, explore trends, and inspect model predictions. SHAP explainability is integrated to provide both global and transaction-level insights, helping risk teams understand why certain transactions are flagged.

---

## Key Results

The Random Forest model trained on e-commerce data achieved very high precision, meaning flagged transactions were almost always fraudulent, but moderate recall, so some fraud cases were missed. This model works best where false positives must be minimized, such as secondary verification layers.

The XGBoost model for credit card transactions captured 81 percent of fraud cases with 66 percent precision and an AUC-PR of 0.81. Only 18 fraud cases were missed, which balances detection effectiveness with manageable false alerts. Evaluation emphasized recall, reflecting the high cost of undetected fraud, while maintaining business-friendly precision levels.

---

## Quick Start

To run the project locally, clone the repository, install dependencies, and launch the dashboard:

```bash
git clone https://github.com/AdeyAmare/financial-fraud-detection
cd financial-fraud-detection
pip install -r requirements.txt
streamlit run app/dashboard.py
```

To run tests:

```bash
pytest tests/
```

---

## Project Structure

The project is organized to separate data processing, feature engineering, modeling, explainability, and application layers. This modular structure ensures reusability and production readiness:

```
financial-fraud-detection/
│
├── data/                  # Raw and processed datasets (excluded from version control)
├── notebooks/             # Exploratory data analysis and feature engineering
├── src/                   # Core pipeline modules
│   ├── data/              # Data loading and preprocessing
│   ├── feature_engineering.py
│   ├── transformation_and_imbalance.py
│   ├── modeling.py
│   ├── explainability.py
│   └── utils/
│
├── models/                # Saved trained models
├── reports/               # Visualizations and evaluation outputs
├── tests/                 # Unit and integration tests
├── app/                   # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Demo

The Streamlit dashboard allows stakeholders to upload transaction data, review fraud trends, compare model outputs, inspect individual transaction scores, and view SHAP-based explanations. This interface presents all insights without technical jargon, making it accessible to risk management and business teams.

[Demo Video](https://streamable.com/vd115n)

---

## Technical Details

### Data

Two datasets were used: an e-commerce dataset containing user, transaction, device, browser, IP, and signup information, and a credit card dataset with anonymized behavioral features. Preprocessing included cleaning, type corrections, IP-to-country mapping, feature scaling, and one-hot encoding of categorical variables. Behavioral features, such as transaction frequency, velocity, hour of day, day of week, and time since signup, were engineered to capture realistic fraud patterns.

Class imbalance, a critical challenge in both datasets, was addressed using SMOTE applied only to the training set to avoid data leakage.

### Model

Logistic Regression was used as a baseline model due to its interpretability. Ensemble models, including Random Forest, XGBoost, and LightGBM, were trained with hyperparameter tuning guided by stratified 5-fold cross-validation. Model performance was evaluated using precision, recall, F1-score, and AUC-PR. Final model selection prioritized business objectives, balancing detection effectiveness with the cost of false alerts.

### SHAP Explainability

SHAP was used to explain model predictions at both global and individual transaction levels. For the e-commerce model, the time since signup was the dominant feature; new accounts strongly increased fraud probability, whereas older accounts were less likely to be flagged. For credit card transactions, features such as V14, V4, V17, and V11 strongly influenced predictions. SHAP values reveal how each feature pushes predictions toward fraud or legitimacy, allowing risk teams to interpret why a transaction is flagged and to act accordingly. Individual SHAP force plots highlight true positives, false positives, and false negatives, providing clear, actionable insights for operational decision-making.

### Evaluation

Models were evaluated using confusion matrices, precision, recall, F1-score, and AUC-PR. Special emphasis was placed on recall due to the high cost of undetected fraud. Cross-validation provided mean and standard deviation metrics for robust performance estimation.

---

## Future Improvements

Future enhancements include cost-sensitive threshold optimization to minimize financial loss, real-time scoring APIs, automated model drift detection and retraining pipelines, fraud risk segmentation, and additional operational dashboards for continuous monitoring. These improvements would move the system closer to enterprise-grade deployment and further reduce the impact of fraudulent transactions.

---

## Author

Adey Amare
- GitHub: [https://github.com/AdeyAmare](https://github.com/AdeyAmare)
- LinkedIn: [https://www.linkedin.com/in/adeyamare/](https://www.linkedin.com/in/adeyamare/)

---

