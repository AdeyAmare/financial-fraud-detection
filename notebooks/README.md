# Fraud Detection Notebooks – Project Overview

This collection of notebooks provides an end-to-end workflow for **fraud detection** using transactional data. The workflow covers raw data exploration, cleaning, geolocation enrichment, feature engineering, data transformation, and modeling for machine learning.

## Notebook Workflow

1. **EDA – Initial Exploration**
   - Load raw fraud or credit card transaction data.
   - Perform initial data quality checks and cleaning (handling missing values, outliers).
   - Conduct univariate and bivariate analysis on key features like purchase value, age, devices, and temporal patterns.
   - Analyze class distribution of fraud vs. legitimate transactions.
   - Save cleaned datasets for downstream processing.

2. **Geolocation Enrichment**
   - Merge fraud transactions with IP-to-country mappings.
   - Validate IP formats and coverage of the mapping dataset.
   - Analyze country-level fraud patterns, including transaction counts and fraud rates.
   - Visualize top countries by fraud rate and transaction volume.
   - Save the geolocation-enriched dataset for feature engineering.

3. **Feature Engineering**
   - Derive behavioral and temporal features:
     - Hour of day, day of week
     - Time since signup
     - Transaction velocity (per user, last 24h)
   - Explore relationships between features and fraud risk using plots and correlation analysis.
   - Visualize fraud trends over time (hourly, daily, weekly).
   - Save the feature-enhanced dataset for modeling.

4. **Data Transformation & Imbalance Handling**
   - Split feature-engineered data into training and test sets.
   - Transform numeric and categorical features as required for modeling.
   - Handle class imbalance in the training set using SMOTE.
   - Visualize the effect of resampling on class distribution.
   - Save the transformed and balanced training dataset for model development.

5. **Modeling**
   - Train and evaluate classification models using:
     - Logistic Regression
     - Random Forest
   - Perform cross-validation and evaluate metrics such as F1, precision, recall, AUC-PR, and confusion matrix.
   - Compare model performance and select the best model for fraud detection.
   - Notebooks:
     - `modeling_credit_card_data.ipynb` – Modeling workflow for credit card transactions
     - `modeling_fraud_data.ipynb` – Modeling workflow for fraud transactions

---

6. **Model Explainability & Interpretation**

This stage focuses on **interpreting model predictions** to understand *why* transactions are classified as fraudulent or legitimate.  
It uses **SHAP (SHapley Additive exPlanations)** to provide both **global** and **local** model insights.

### Objectives

- Identify which features most strongly influence fraud predictions.
- Explain individual fraud decisions for transparency and trust.
- Analyze model behavior on:
  - Correct fraud detections
  - False alarms
  - Missed fraud cases
- Translate model outputs into **actionable business insights**.

### Explainability Workflow

1. **Load Trained Artifacts**
   - Load the saved best-performing model.
   - Load the corresponding preprocessing pipeline.
   - Apply preprocessing to the test dataset only.

2. **Global Model Interpretation**
   - Plot **built-in feature importance** (for tree-based models).
   - Generate a **SHAP summary plot** to show overall feature impact.
   - Identify dominant behavioral, temporal, and transaction-related drivers of fraud risk.

3. **Local Prediction Explanations**
   - Generate **SHAP force plots** for individual cases:
     - **True Positive (TP):** Correctly detected fraud
     - **False Positive (FP):** Legitimate transaction flagged as fraud
     - **False Negative (FN):** Fraud transaction missed by the model
   - Analyze feature contributions for each decision.

4. **Top Fraud Drivers**
   - Extract the most influential features using **mean absolute SHAP values**.
   - Compare model-driven risk factors with domain expectations.

### Notebook

- `explainability_fraud_data.ipynb` and `explainability_credit_data.ipynb`
  - Loads trained models and preprocessors
  - Produces SHAP summary plots and force plots
  - Documents insights and business implications

### Key Outputs

- Global SHAP summary plots (feature importance)
- Instance-level SHAP force plots (decision explanations)
- Ranked list of top fraud-driving features
- Interpretability insights supporting model evaluation and deployment decisions

### Design Notes

- Explainability is performed **post-training** and does not affect model performance.
- Uses defensive handling for SHAP dimensional edge cases.
- Supports both:
  - Tree-based models (Random Forest, Gradient Boosting)
  - Linear / non-tree models (via model-agnostic SHAP)
- Intended for **analysis, reporting, and stakeholder communication**.

---


## Key Goals

- Understand patterns in fraudulent and legitimate transactions.
- Enrich data with geolocation for more granular risk analysis.
- Create informative features that improve predictive modeling.
- Prepare a clean, balanced dataset ready for machine learning.
- Evaluate models to identify the most effective classifier.

## Data Output

- `fraud_data_cleaned.csv` – Cleaned raw data  
- `fraud_data_with_country.csv` – Geolocation-enriched data  
- `fraud_data_with_features.csv` – Feature-engineered dataset  
- `fraud_train_smote.csv` – Transformed and SMOTE-balanced training dataset  
- Model artifacts and evaluation results are stored as outputs from the modeling notebooks.

## Usage

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
````

2. **Run notebooks in order**:

   1. `eda_fraud_data.ipynb` and `eda_credit_card.ipynb` – Initial exploration and cleaning
   2. `ipaddress_to_country.ipynb` – Merge country info
   3. `feature_engineering.ipynb` – Create behavioral and temporal features
   4. `data_transformation_imbalance_handling.ipynb` – Transform features, handle imbalance
   5. `modeling_credit_card_data.ipynb` and `modeling_fraud_data.ipynb` – Train, evaluate, and select ML models

3. **Use the output datasets** for further machine learning or analysis.

4. **Adjust paths** in notebooks if your folder structure differs.

## Notes

* All visualizations are intended for exploratory insight.
* SMOTE is applied only to the training set to prevent data leakage.
* Modeling notebooks rely on feature-engineered and transformed datasets.

