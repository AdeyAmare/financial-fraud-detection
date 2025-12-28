
# Improved detection of fraud cases for e-commerce and bank transactions

This project provides a complete pipeline for **fraud detection** using transaction data. It includes data exploration, cleaning, geolocation enrichment, feature engineering, data transformation, and preparation for machine learning modeling.

---

## Project Structure

```

project/
├── data/                  # Raw and processed datasets (ignored by .gitignore)
├── notebooks/             # EDA, feature engineering, transformation, and modeling notebooks
├── src/                   # Source code (loaders, cleaners, feature engineering, transformers, modeling, utils)
├── tests/                 # Unit and integration tests for pipeline components
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

5. **Modeling**  
   Train, evaluate, and compare classification models:
   - Logistic Regression  
   - Random Forest  
   Metrics include F1, precision, recall, AUC-PR, and confusion matrix.

**Notebook Order:**  
See the [Notebooks README](./notebooks/README.md) for details, usage instructions, and visualization outputs.

---

## Source Code (`src/`)

All reusable code is in the `src` folder:

- **Data utilities (`src/data`)**: Loaders, cleaners, and IP-to-country merger ([README](./src/data/README.md))  
- **Feature engineering (`feature_engineering.py`)**: Temporal and behavioral features  
- **Transformation & imbalance (`transformation_and_imbalance.py`)**: Scaling, encoding, SMOTE  
- **Modeling pipeline (`modeling.py`)**: Logistic Regression and Random Forest pipeline with evaluation, cross-validation, and model comparison  
- **Utilities (`src/utils/io_utils.py`)**: Safe CSV saving ([README](./src/utils/README.md))

**Design Goals:**
- Modular and reusable code for ETL, feature engineering, preprocessing, and modeling
- Logging-enabled for debugging and reproducibility
- Outputs ready for ML model training

---

## Tests (`tests/`)

- Contains **unit and integration tests** for the pipeline components
- Validates data loaders, transformers, feature engineering, and modeling pipeline
- Ensures reproducibility and correctness of the workflow
- Run tests with:
```bash
pytest tests/
```

---

## Data Outputs

* `fraud_data_cleaned.csv` – Cleaned raw data
* `fraud_data_with_country.csv` – Geolocation-enriched data
* `fraud_data_with_features.csv` – Feature-engineered dataset
* `fraud_train_smote.csv` – Transformed and SMOTE-balanced training dataset
* Model artifacts and evaluation results are generated in modeling notebooks

---

## Usage

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Run notebooks in order** to replicate the full workflow. Refer to [Notebooks README](./notebooks/README.md).

3. **Use `src` modules** for programmatic access to loaders, cleaners, feature engineering, transformations, and modeling:

```python
from src.data.loader import FraudDataLoader
from src.feature_engineering import FraudFeatureEngineer
from src.transformation_and_imbalance import FraudDataTransformer
from src.modeling_pipeline import ModelingPipeline
from src.utils.io_utils import save_dataframe
```

4. **Run tests** to validate pipeline functionality:

```bash
pytest tests/
```

5. **Adjust file paths** if your folder structure differs.

---

## Notes

* SMOTE is applied only on the training set to prevent data leakage.
* IP-to-country coverage affects geolocation enrichment accuracy.
* All visualizations are exploratory and intended for insight into fraud patterns.
* Modeling notebooks rely on feature-engineered and transformed datasets.

