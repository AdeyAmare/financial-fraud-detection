# Fraud Detection Notebooks – Project Overview

This collection of notebooks provides a complete workflow for fraud detection using e-commerce and credit card transaction data. It covers raw data exploration, cleaning, geolocation enrichment, feature engineering, transformation and imbalance handling, model training, evaluation, and explainability. Each notebook is designed to produce reproducible outputs suitable for downstream modeling, analysis, and stakeholder reporting.

---

## Notebook Workflow

The workflow is structured to follow a logical progression from raw data to deployable machine learning models.

### EDA – Initial Exploration

The first step involves loading raw fraud or credit card transaction data and performing initial quality checks. This includes handling missing values, removing duplicates, and inspecting outliers. Univariate and bivariate analyses are conducted on features such as purchase amount, user age, device usage, and temporal patterns. Class distribution of fraud versus legitimate transactions is quantified. Cleaned datasets are saved for further processing.

### Geolocation Enrichment

Fraud transactions are enriched with country-level information using IP-to-country mapping. The notebooks validate IP formats and coverage, ensuring robust merges. Country-level fraud statistics are computed, including total transactions, fraud counts, and fraud rates. Top countries by transaction volume and fraud prevalence are visualized. The resulting geolocation-enriched dataset is saved for feature engineering.

### Feature Engineering

Behavioral and temporal features are derived to enhance predictive power. Hour of the day, day of the week, time since signup, and transaction velocity over the last 24 hours are added. Relationships between features and fraud risk are visualized and analyzed, including correlation analysis and temporal fraud trends. The feature-enhanced dataset is saved for modeling.

### Data Transformation and Imbalance Handling

Feature-engineered data is split into training and test sets using stratified sampling to preserve class distributions. Numeric and categorical features are transformed appropriately for model consumption. Class imbalance is addressed using SMOTE on the training set. The effects of resampling on class distribution are visualized and documented. Transformed and balanced training datasets are saved for model development.

### Modeling

Classification models are trained and evaluated on the prepared datasets. Logistic Regression provides an interpretable baseline, while Random Forest captures complex patterns in fraud behavior. Cross-validation with Stratified K-Fold is used to report F1-score, precision, recall, AUC-PR, and confusion matrices. Model comparisons inform selection of the best-performing classifier for both e-commerce and credit card fraud detection. Specific modeling notebooks include `credit_data_modeling.ipynb` for credit card transactions and `fraud_data_modeling.ipynb` for e-commerce transactions.

---

## Model Explainability and Interpretation

Model explainability focuses on understanding why transactions are flagged as fraudulent or legitimate, supporting transparency and actionable insights. SHAP (SHapley Additive exPlanations) is used for both global and local interpretability.

### Objectives

The explainability step identifies features that strongly influence model predictions, explains individual transaction decisions, and highlights areas where models may over-flag or miss fraud. The analysis provides insights into correct fraud detection, false positives, and false negatives.

### Explainability Workflow

Trained models and preprocessors are loaded and applied to the test dataset. Global interpretation includes plotting built-in feature importance and generating SHAP summary plots to visualize feature influence across the dataset. Local interpretation uses SHAP force plots to explain individual transactions, focusing on true positives, false positives, and false negatives. The most influential features are extracted using mean absolute SHAP values, providing a ranked list of top fraud drivers. These outputs enable risk teams and business stakeholders to understand model behavior and inform operational decisions.

### Notebooks

`explainability_fraud_data.ipynb` and `explainability_credit_data.ipynb` perform post-training analysis, produce SHAP visualizations, and document business-relevant insights.

### Key Outputs

Global SHAP summary plots highlight features with the largest impact on fraud predictions. Instance-level SHAP force plots explain individual transaction decisions. A ranked list of top fraud drivers is generated to support interpretability and policy recommendations. This stage is performed post-training and does not affect model performance.

---

## Key Goals

The notebook workflow is designed to provide a complete fraud detection solution. It enables understanding patterns in both fraudulent and legitimate transactions, enriches data with geolocation for finer-grained risk assessment, creates informative behavioral and temporal features, prepares balanced datasets for modeling, and evaluates classifiers to identify the most effective models.

---

## Usage

First, install project dependencies:

```bash
pip install -r requirements.txt
```

Run notebooks in order:

1. `eda_fraud_data.ipynb` and `eda_credit_card.ipynb` perform initial exploration and cleaning.
2. `ipaddress_to_country.ipynb` merges country information with transactions.
3. `feature_engineering.ipynb` adds behavioral and temporal features.
4. `data_transformation_imbalance_handling.ipynb` transforms features and handles class imbalance.
5. `credit_data_modeling.ipynb` and `fraud_data_modeling.ipynb` train, evaluate, and select ML models.

Output datasets can be reused for further modeling, analysis, or integration into dashboards. Adjust notebook paths as needed to match your project folder structure.

---

## Notes

Visualizations are designed for exploratory and business insight. SMOTE oversampling is applied only to training sets to avoid data leakage. Modeling relies on feature-engineered and transformed datasets. SHAP explainability is performed post-training to ensure interpretability without affecting model predictions.
