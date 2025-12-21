# Tests

This folder contains unit tests for the **Fraud Detection Project**, covering data loading, cleaning, geolocation merging, feature engineering, and data transformation pipelines.

---

## Test Files

### 1. `test_data_utils.py`
- Tests the core data processing components:
  - `TransactionDataCleaner`: duplicate removal, missing value handling, data type corrections.
  - Data loaders:
    - `FraudDataLoader`
    - `CreditCardDataLoader`
    - `IPCountryLoader`
  - `GeoDataMerger`: IP-to-country conversion, merging, and fraud statistics computation.

### 2. `test_feature_engineering.py`
- Tests the `FraudFeatureEngineer` class:
  - Timestamp parsing
  - Time-based features: `hour_of_day`, `day_of_week`, `time_since_signup`
  - Transaction velocity: `transactions_per_user`, `transactions_last_24h`
  - Ensures all expected features are created correctly.

### 3. `test_data_transformer.py`
- Tests the `FraudDataTransformer` class:
  - Train/test split
  - Numeric scaling and categorical encoding
  - Handling class imbalance using SMOTE
  - Verifies consistency of shapes and feature encoding.

---

## How to Run Tests

1. Install `pytest` if not already installed:

```bash
pip install pytest
````

2. From the project root, run all tests:

```bash
pytest tests/
```

3. To run a specific test file:

```bash
pytest tests/test_feature_engineering.py
```

---

## Notes

* Tests use `pytest` fixtures for sample data.
* Temporary CSV files are created using `tmp_path` for loader tests.
* These tests validate functionality **without needing full datasets**.
* Designed to catch regressions in data handling, feature engineering, and preprocessing.

```

