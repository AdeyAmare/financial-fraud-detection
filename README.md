# Improved detection of fraud cases for e-commerce and bank transactions

This project provides a complete pipeline for **fraud detection** using transaction data. It includes data exploration, cleaning, geolocation enrichment, feature engineering, data transformation, and preparation for machine learning modeling.

---

## Project Structure

```

project/
├── data/                  # Raw and processed datasets (ignored by .gitignore)
├── notebooks/             # EDA, feature engineering, transformation notebooks
├── src/                   # Source code (loaders, cleaners, feature engineering, transformers, utils)
├── requirements.txt       # Python dependencies
└── README.md              # Project-level documentation

```

---

## Notebooks Workflow

The notebooks provide an **end-to-end reproducible workflow**:

1. **EDA & Initial Cleaning**  
   Explore datasets, handle missing values and duplicates, visualize distributions, temporal patterns, and class imbalance. Save cleaned datasets.

2. **Geolocation Enrichment**  
   Merge IP-to-country mapping with transactions and analyze fraud patterns per country.

3. **Feature Engineering**  
   Create temporal and behavioral features:
   - Hour of day, day of week  
   - Time since signup  
   - Transaction velocity (per user, last 24h)

4. **Data Transformation & Imbalance Handling**  
   Split data into training and test sets, scale numeric features, encode categorical features, and apply SMOTE.

**Notebook Order:**  
See the [Notebooks README](./notebooks/README.md) for details, usage instructions, and visualization outputs.

---

## Source Code (`src/`)

All reusable code is in the `src` folder:

- **Data utilities (`src/data`)**: Loaders, cleaners, and IP-to-country merger ([README](./src/data/README.md))  
- **Feature engineering (`feature_engineering.py`)**: Temporal and behavioral features  
- **Transformation & imbalance (`transformation_and_imbalance.py`)**: Scaling, encoding, SMOTE  
- **Utilities (`src/utils/io_utils.py`)**: Safe CSV saving ([README](./src/utils/README.md))

**Design Goals:**
- Modular and reusable code for ETL, feature engineering, and preprocessing
- Logging-enabled for debugging and reproducibility
- Outputs ready for ML model training

---

## Data Outputs

- `fraud_data_cleaned.csv` – Cleaned raw data  
- `fraud_data_with_country.csv` – Geolocation-enriched data  
- `fraud_data_with_features.csv` – Feature-engineered dataset  
- `fraud_train_smote.csv` – Transformed and SMOTE-balanced training dataset  

---

## Usage

1. **Install dependencies**
```bash
pip install -r requirements.txt
````

2. **Run notebooks in order** to replicate the full workflow. Refer to [Notebooks README](./notebooks/README.md).

3. **Use `src` modules** for programmatic access to loaders, cleaners, feature engineering, and transformations:

```python
from src.data.loader import FraudDataLoader
from src.feature_engineering import FraudFeatureEngineer
from src.transformation_and_imbalance import FraudDataTransformer
from src.utils.io_utils import save_dataframe
```

4. **Adjust file paths** if your folder structure differs.

---

## Notes

* SMOTE is applied only on the training set to prevent data leakage.
* IP-to-country coverage affects geolocation enrichment accuracy.
* All visualizations are exploratory and intended for insight into fraud patterns.
