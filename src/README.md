# `src` – Source Code for Fraud Detection Project

This folder contains the core Python modules for the fraud detection project, including data loading, cleaning, geolocation merging, feature engineering, transformation, imbalance handling, modeling pipelines, and explainability. The code is structured for reproducible ETL workflows, modular experimentation, and production deployment.

---

## Folder Structure

```
src/
├── data/                                # Loaders, cleaners, and geolocation merger utilities
│   ├── loader.py
│   ├── cleaner.py
│   └── merger.py
├── feature_engineering.py               # Temporal and behavioral feature engineering
├── transformation_and_imbalance.py      # Feature scaling, encoding, and SMOTE handling
├── modeling.py                          # Modeling pipeline for classification and evaluation
├── explainability.py                     # SHAP-based global and local explainability
└── utils/
    └── io_utils.py                      # Safe CSV I/O utilities
```

---

## 1. Data Utilities (`src/data`)

This folder contains reusable classes for loading, cleaning, and enriching transactional datasets. It supports both e-commerce fraud transactions and bank credit card datasets.

The `FraudDataLoader`, `CreditCardDataLoader`, and `IPCountryLoader` classes standardize CSV ingestion and ensure validation of required columns. The `TransactionDataCleaner` class handles duplicates, missing values, and type corrections in a reproducible pipeline. The `GeoDataMerger` class enables IP-to-country enrichment, efficiently mapping IP ranges and computing country-level fraud statistics.

Refer to the [data README](./data/README.md) for complete details and usage examples.

---

## 2. Feature Engineering (`feature_engineering.py`)

The `FraudFeatureEngineer` class is responsible for creating temporal and behavioral features that capture patterns indicative of fraud. This includes parsing timestamps, adding hour-of-day and day-of-week features, computing the time elapsed since account signup, and calculating transaction velocity for each user.

This module allows chaining of methods to build a feature-rich DataFrame ready for model input. It is designed for modular use in notebooks and production pipelines.

Usage Example:

```python
from src.feature_engineering import FraudFeatureEngineer

engineer = FraudFeatureEngineer(fraud_df)
engineer.parse_timestamps().add_time_features().add_time_since_signup().add_transaction_velocity()
features_df = engineer.get_features()
```

---

## 3. Transformation and Imbalance Handling (`transformation_and_imbalance.py`)

`FraudDataTransformer` prepares datasets for machine learning. It handles train-test splitting with stratification to preserve class distribution, numeric feature scaling, categorical feature encoding, and oversampling of minority classes using SMOTE.

The module ensures that all transformations are applied consistently and can return processed training and testing sets ready for modeling. This approach isolates data preparation from modeling logic, promoting reproducibility and reducing leakage risks.

Usage Example:

```python
from src.transformation_and_imbalance import FraudDataTransformer

transformer = FraudDataTransformer(df, target='class', numeric_features=numeric_cols, categorical_features=categorical_cols)
transformer.split_data().transform_features().handle_imbalance('SMOTE')
X_train, X_test, y_train, y_test = transformer.get_train_test()
```

---

## 4. Modeling Pipeline (`modeling.py`)

`ModelingPipeline` provides end-to-end model training, evaluation, and comparison. It supports Logistic Regression and Random Forest models, with optional SMOTE oversampling. Preprocessing steps such as scaling and encoding are integrated to ensure consistency.

The module implements cross-validation using Stratified K-Fold, reporting metrics including F1-score, precision, recall, AUC-PR, and confusion matrix. It also provides utilities to compare models and select the best-performing model based on both statistical metrics and business considerations.

Usage Example:

```python
from src.modeling import ModelingPipeline

numeric_cols = ["Time", "Amount"]
categorical_cols = []

pipeline = ModelingPipeline(df, numeric_features=numeric_cols, categorical_features=categorical_cols, target_col='Class')
pipeline.prepare_data()
lr_model = pipeline.train_logistic_regression()
rf_model = pipeline.train_random_forest()
comparison = pipeline.compare_models()
best_model, justification = pipeline.select_best_model()
```

---

## 5. Model Explainability (`explainability.py`)

The `ModelExplainability` class leverages SHAP (SHapley Additive exPlanations) for both global and local interpretability of fraud detection models. It is compatible with saved models and preprocessors, enabling post-hoc explainability without retraining.

Global explainability identifies the most influential features across the dataset, while local explanations visualize why individual transactions were classified as fraud or legitimate. Special handling is included for true positives, false positives, and false negatives. SHAP outputs, combined with built-in model importance, support business insights and operational decisions.

Usage Example:

```python
from src.explainability import ModelExplainability

explainer = ModelExplainability(
    model_path="models/fraud_model.joblib",
    preprocessor_path="models/preprocessor.joblib",
    X=X_test,
    y=y_test
)

explainer.plot_builtin_feature_importance(top_n=10)
explainer.plot_shap_summary()
explainer.plot_force_plot_for_case("TP")
explainer.plot_force_plot_for_case("FP")
explainer.plot_force_plot_for_case("FN")
top_features = explainer.get_top_drivers(top_n=5)
```

---

## 6. Utilities (`src/utils`)

`io_utils.py` provides safe data input/output operations, such as saving DataFrames to CSV. It ensures that directories exist before writing and prevents accidental overwrites.

Usage Example:

```python
from src.utils.io_utils import save_dataframe

save_dataframe(features_df, "data/processed/fraud_data_with_features.csv")
```

---

## Notes and Best Practices

All modules use logging to track processing steps and potential issues. The workflow is designed to follow a **modular and reproducible pipeline**: load and clean data, enrich with geolocation, engineer features, transform and balance the dataset, train and evaluate models, and finally interpret predictions.

This structure ensures that outputs, including transformed datasets and trained models, are immediately ready for integration into **production ML workflows**, dashboards, or downstream analytics.
