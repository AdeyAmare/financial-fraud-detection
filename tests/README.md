# Tests – Fraud Detection Project

The `tests` folder contains unit tests for all major components of the Fraud Detection Project, including **data loading, cleaning, geolocation merging, feature engineering, transformation, and modeling pipelines**. These tests are designed to validate functionality, catch regressions, and ensure the project runs correctly without requiring the full datasets.

## Test Coverage

Tests include data utilities such as `TransactionDataCleaner` for duplicate removal, missing value handling, and type corrections. Data loaders like `FraudDataLoader`, `CreditCardDataLoader`, and `IPCountryLoader` are verified for correct file loading and validation. `GeoDataMerger` is tested for IP-to-country conversion, proper merging, and computation of country-level fraud statistics.

Feature engineering is validated through tests of the `FraudFeatureEngineer` class, ensuring timestamp parsing, extraction of time-based features such as hour-of-day, day-of-week, and time-since-signup, as well as calculation of transaction velocity per user and rolling 24-hour counts. Tests confirm that all expected features are created correctly.

Data transformation pipelines are tested through the `FraudDataTransformer`, which handles stratified train-test splitting, numeric scaling, categorical encoding, and class imbalance using SMOTE. Tests verify consistency in shapes and correct handling of resampled data.

The `ModelingPipeline` is now fully covered by a dedicated end-to-end test using synthetic data. This test ensures that data preparation, logistic regression training, random forest training, model comparison, and best model selection run without errors. Synthetic datasets are generated with numeric and categorical columns and a binary target. Logging and assertions are included to track progress and validate each step.

---

## Running Tests

To run the tests, first ensure that `pytest` is installed. From the project root, you can execute all tests using `pytest tests/`. Specific test files can be run individually by specifying their path, for example `pytest tests/test_feature_engineering.py`. The tests rely on lightweight synthetic or temporary data, meaning the full datasets are not required for execution.

---

## Notes

The tests use `pytest` fixtures and temporary file handling to simulate realistic scenarios for loaders and transformers. Logging provides insight into progress and test validation. Warnings from underlying libraries may appear but are not indicative of test failures.

The addition of the `ModelingPipeline` end-to-end test ensures that the modeling workflow can run fully, including data preparation, model training, evaluation, comparison, and best model selection, all using reproducible synthetic data. This makes the tests suitable for integration into a CI/CD workflow or automated regression testing.

---

