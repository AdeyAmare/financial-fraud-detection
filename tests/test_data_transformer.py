import pytest
import pandas as pd
import logging
from dataclasses import dataclass
from typing import List
from src.transformation_and_imbalance import FraudDataTransformer

# -----------------------------
# Constants
# -----------------------------
NUMERIC_FEATURES: List[str] = ["purchase_value", "age"]
CATEGORICAL_FEATURES: List[str] = ["sex", "browser"]
TARGET_COLUMN: str = "class"
SAMPLE_SIZE: int = 20

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# Dataclasses
# -----------------------------
@dataclass
class SampleFraudData:
    """
    Dataclass to structure sample data for FraudDataTransformer tests.
    """
    purchase_value: List[int]
    age: List[int]
    sex: List[str]
    browser: List[str]
    clazz: List[int]  # renamed from 'class' to avoid Python keyword conflict

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataclass to a pandas DataFrame.
        """
        return pd.DataFrame({
            "purchase_value": self.purchase_value,
            "age": self.age,
            "sex": self.sex,
            "browser": self.browser,
            "class": self.clazz
        })

# -----------------------------
# Utility Functions
# -----------------------------
def create_sample_dataframe() -> pd.DataFrame:
    """
    Creates a sample DataFrame for testing the fraud data transformer.

    Returns:
        pd.DataFrame: Sample data including numeric, categorical features and target column.
    """
    logger.info("Creating sample DataFrame with %d rows", SAMPLE_SIZE)
    data = SampleFraudData(
        purchase_value=[100 + i * 10 for i in range(SAMPLE_SIZE)],
        age=[20 + i for i in range(SAMPLE_SIZE)],
        sex=["M", "F"] * (SAMPLE_SIZE // 2),
        browser=["Chrome", "Safari", "Firefox", "Edge", "Opera"] * (SAMPLE_SIZE // 5),
        clazz=[0, 1] * (SAMPLE_SIZE // 2)
    )
    df = data.to_dataframe()
    logger.debug("Sample DataFrame created:\n%s", df.head())
    return df

def validate_shapes(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
    """
    Validates the basic shapes of train and test datasets.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Testing target.

    Raises:
        AssertionError: If shapes are inconsistent or invalid.
    """
    logger.info("Validating train/test dataset shapes")
    assert X_train.shape[0] == len(y_train), "Mismatch in number of training samples"
    assert X_test.shape[0] == len(y_test), "Mismatch in number of test samples"
    assert X_train.shape[1] > 0, "No features present in training data"
    assert X_test.shape[1] == X_train.shape[1], "Mismatch in number of features between train and test"

# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Pytest fixture to provide a sample DataFrame.
    """
    return create_sample_dataframe()

# -----------------------------
# Test Function
# -----------------------------
def test_fraud_data_transformer_pipeline(sample_df: pd.DataFrame) -> None:
    """
    Tests the end-to-end fraud data transformation pipeline including:
        - Train/test split
        - Feature transformation (numeric + categorical)
        - Imbalance handling

    Args:
        sample_df (pd.DataFrame): Sample dataset provided by fixture.
    """
    logger.info("Initializing FraudDataTransformer pipeline")
    transformer = FraudDataTransformer(
        df=sample_df,
        target=TARGET_COLUMN,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES
    )

    # Run the full pipeline
    transformer.split_data().transform_features().handle_imbalance()
    X_train, X_test, y_train, y_test = transformer.get_train_test()
    
    # Validate dataset shapes
    validate_shapes(X_train, X_test, y_train, y_test)
    logger.info("FraudDataTransformer pipeline test passed successfully")
