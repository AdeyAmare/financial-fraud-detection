# `src/data` – Data Loading and Cleaning Utilities

This folder contains Python modules for **loading, cleaning, and merging transaction datasets**, including fraud and credit card data.

## Modules

### 1. `loader.py`
Provides classes to load different datasets into pandas DataFrames:

- **`FraudDataLoader`**
  - Loads raw fraud transaction CSVs.
  - Ensures required columns `signup_time` and `purchase_time` exist.
  - Converts date columns to datetime objects.
  
- **`CreditCardDataLoader`**
  - Loads credit card transaction CSVs.
  - Simple wrapper around `pandas.read_csv`.

- **`IPCountryLoader`**
  - Loads IP-to-country mapping CSVs.
  - Ensures required columns: `lower_bound_ip_address`, `upper_bound_ip_address`, `country`.

**Usage Example:**
    
```python
    from src.data.loader import FraudDataLoader

    loader = FraudDataLoader("data/raw/Fraud_Data.csv")
    df = loader.load()
```

### 2. `cleaner.py`

Provides the **`TransactionDataCleaner`** class for cleaning transactional datasets:

* Reports dataset statistics: shape, dtypes, duplicates, missing values.
* Removes duplicate rows.
* Handles missing values with multiple strategies:

  * `drop` – drop rows with missing values
  * `fill_zero` – fill missing values with 0
  * `fill_median` – fill numeric columns with median
* Corrects data types (e.g., ensures `age` is integer).
* Offers a full `clean()` pipeline combining all steps.

**Usage Example:**

```python
from src.data.cleaner import TransactionDataCleaner

cleaner = TransactionDataCleaner(df)
cleaner.report()
df_clean = cleaner.clean(missing_strategy="fill_median")
```

### 3. `merger.py`

Provides the **`GeoDataMerger`** class for IP-to-country geolocation enrichment:

* Converts IP addresses to integers for range comparison.
* Prepares and sorts IP ranges from mapping data.
* Merges transaction data with IP ranges using `merge_asof`.
* Assigns "Unknown" for unmapped or invalid IPs.
* Computes country-level fraud statistics:

  * `total_transactions`
  * `fraud_count`
  * `fraud_rate`

**Usage Example:**

```python
from src.data.loader import FraudDataLoader, IPCountryLoader
from src.data.merger import GeoDataMerger

fraud_df = FraudDataLoader("data/processed/fraud_data_cleaned.csv").load()
ip_df = IPCountryLoader("data/raw/IpAddress_to_Country.csv").load()

merger = GeoDataMerger(fraud_df, ip_df)
fraud_df_geo = merger.merge_country()
summary = merger.get_summary()
```

---

## Notes

* All loaders validate the existence of the file and required columns.
* The cleaner is designed to handle typical transaction dataset issues, but can be extended for custom transformations.
* The merger assumes numeric or string IP addresses in standard IPv4 format.
* Designed for **reproducible, modular ETL pipelines** in fraud detection projects.

