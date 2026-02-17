# `src/data` – Data Loading, Cleaning, and Geolocation Utilities

This module folder provides reusable Python classes and functions for **loading, cleaning, and merging transactional datasets**, specifically designed for fraud detection tasks in e-commerce and banking contexts. It supports both raw fraud transaction datasets and credit card datasets, and includes IP-to-country geolocation enrichment.

All components are designed for modular ETL pipelines, ensuring reproducibility, maintainability, and easy integration with modeling workflows.

---

## Modules Overview

### `loader.py`

This module contains dataset loaders that standardize the process of reading CSV files and converting them into clean pandas DataFrames. Each loader validates required columns and performs minimal preprocessing to prepare the data for downstream processing.

`FraudDataLoader` focuses on e-commerce transaction data. It ensures that critical columns such as `signup_time` and `purchase_time` exist and converts them to datetime objects to facilitate time-based feature engineering.

`CreditCardDataLoader` loads bank credit card transaction data. It provides a simple wrapper around `pandas.read_csv` while ensuring type consistency for numeric and categorical columns.

`IPCountryLoader` loads IP-to-country mapping datasets. It validates that the CSV contains `lower_bound_ip_address`, `upper_bound_ip_address`, and `country`. These mappings are later used for geolocation enrichment.

Usage example:

```python
from src.data.loader import FraudDataLoader, IPCountryLoader

fraud_loader = FraudDataLoader("data/raw/Fraud_Data.csv")
fraud_df = fraud_loader.load()

ip_loader = IPCountryLoader("data/raw/IpAddress_to_Country.csv")
ip_df = ip_loader.load()
```

---

### `cleaner.py`

This module provides the `TransactionDataCleaner` class for robust cleaning of transactional datasets. It combines reporting, type correction, duplicate removal, and missing value handling in a single, reproducible pipeline.

The cleaner can generate detailed reports about dataset shape, data types, missing values, and duplicates. Missing values can be handled using several strategies, including dropping rows, filling with zeros, or filling numeric columns with the median value. Data types are also corrected, for example converting `age` to integer or ensuring timestamps are datetime objects.

The `clean()` method executes all cleaning steps in sequence, allowing easy integration into preprocessing pipelines for modeling.

Usage example:

```python
from src.data.cleaner import TransactionDataCleaner

cleaner = TransactionDataCleaner(fraud_df)
cleaner.report()
fraud_df_cleaned = cleaner.clean(missing_strategy="fill_median")
```

---

### `merger.py`

The `GeoDataMerger` class enriches transaction datasets with geolocation information based on IP-to-country mappings. It converts IP addresses to integer format for range-based joins, sorts IP ranges, and performs a `merge_asof` to efficiently map IP addresses to countries.

Transactions with unmapped or invalid IPs are assigned "Unknown" to ensure completeness. The module can compute country-level fraud statistics, including total transactions, number of frauds, and fraud rates, which are essential for exploratory analysis and reporting.

Usage example:

```python
from src.data.loader import FraudDataLoader, IPCountryLoader
from src.data.merger import GeoDataMerger

fraud_df = FraudDataLoader("data/processed/fraud_data_cleaned.csv").load()
ip_df = IPCountryLoader("data/raw/IpAddress_to_Country.csv").load()

merger = GeoDataMerger(fraud_df, ip_df)
fraud_df_geo = merger.merge_country()
summary_stats = merger.get_summary()
```

---

## Best Practices

All loaders, cleaners, and mergers are designed to be modular, allowing them to be integrated into larger pipelines for feature engineering, modeling, and evaluation.

Loaders always validate input files and required columns. The cleaner provides configurable missing value handling strategies and type corrections. The merger is optimized for large datasets, using integer-based IP joins to handle millions of transactions efficiently.

This structure ensures that ETL operations in fraud detection projects are **reproducible, transparent, and production-ready**, with minimal manual intervention.

---