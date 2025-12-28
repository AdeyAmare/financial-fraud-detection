# `src` – Source Code for Fraud Detection Project

This folder contains all source code for the fraud detection project, including **data loading and cleaning, geolocation merging, feature engineering, data transformation, and modeling pipelines**.

## Folder Structure

```

src/
├── data/                     # Data loaders, cleaners, and merger utilities
│   ├── loader.py
│   ├── cleaner.py
│   └── merger.py
├── feature_engineering.py     # Temporal and behavioral feature engineering
├── transformation_and_imbalance.py  # Feature transformation, scaling, encoding, SMOTE
├── modeling.py       # Modeling pipeline with Logistic Regression & Random Forest
└── utils/
└── io_utils.py           # Data I/O utilities (CSV save)

```

## 1️⃣ Data Utilities (`src/data`)

Contains **loaders, cleaners, and merger utilities** for fraud and credit card datasets.

For detailed documentation, see: [src/data README](./data/README.md)

**Highlights:**
- `FraudDataLoader`, `CreditCardDataLoader`, `IPCountryLoader` – CSV loaders with validation
- `TransactionDataCleaner` – Remove duplicates, handle missing values, correct data types
- `GeoDataMerger` – Merge IP address ranges with transactions and compute country-level fraud statistics

---

## 2️⃣ Feature Engineering (`feature_engineering.py`)

Provides the **`FraudFeatureEngineer`** class for creating temporal and behavioral features:

- `parse_timestamps()` – Convert signup and purchase times to datetime
- `add_time_features()` – Extract `hour_of_day` and `day_of_week`
- `add_time_since_signup()` – Compute hours since signup
- `add_transaction_velocity()` – Calculate per-user total transactions and 24h rolling counts
- `get_features()` – Return the feature-enhanced DataFrame

**Usage Example:**
```python
from src.feature_engineering import FraudFeatureEngineer

engineer = FraudFeatureEngineer(fraud_df)
engineer.parse_timestamps().add_time_features().add_time_since_signup().add_transaction_velocity()
features_df = engineer.get_features()
```

---

## 3️⃣ Transformation & Imbalance (`transformation_and_imbalance.py`)

Provides **`FraudDataTransformer`** to prepare data for modeling:

* `split_data()` – Stratified train/test split
* `transform_features()` – Standard scale numeric, one-hot encode categorical features
* `handle_imbalance()` – Apply SMOTE to training data
* `get_train_test()` – Retrieve processed training and test sets

**Usage Example:**

```python
from src.transformation_and_imbalance import FraudDataTransformer

transformer = FraudDataTransformer(df, target='class', numeric_features=numeric_cols, categorical_features=categorical_cols)
transformer.split_data().transform_features().handle_imbalance('SMOTE')
X_train, X_test, y_train, y_test = transformer.get_train_test()
```

---

## 4️⃣ Modeling Pipeline (`modeling.py`)

Provides the **`ModelingPipeline`** class for training, evaluating, and comparing classification models:

* Supports **Logistic Regression** and **Random Forest**.
* Optional **SMOTE** oversampling to handle class imbalance.
* Handles **preprocessing** (standard scaling numeric features, one-hot encoding categorical features).
* **Cross-validation** (Stratified K-Fold) with metrics: F1, AUC-PR, precision, recall, confusion matrix.
* Methods for **comparing models** and **selecting the best model** based on evaluation metrics.

**Usage Example:**

```python
from src.modeling_pipeline import ModelingPipeline

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

## 5️⃣ Utilities (`src/utils`)

Provides helper functions for **data input/output**:

* `save_dataframe(df, path, index=False)` – Save DataFrame to CSV safely, creating directories if needed

For detailed documentation, see: [src/utils README](./utils/README.md)

**Usage Example:**

```python
from src.utils.io_utils import save_dataframe

save_dataframe(features_df, "data/processed/fraud_data_with_features.csv")
```

---

## Notes

* All modules use **logging** to track progress and issues.
* Designed for modular, reproducible ETL, feature, and modeling pipelines.
* Follow the **notebook workflow** in order:

  1. Data loading & cleaning
  2. Geolocation enrichment
  3. Feature engineering
  4. Data transformation & imbalance handling
  5. Model training, evaluation, and selection
* Output datasets and trained models are ready for **machine learning workflows**.


