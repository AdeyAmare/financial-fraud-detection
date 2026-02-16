import pytest
import pandas as pd
import logging
from dataclasses import dataclass
from typing import List

from src.feature_engineering import FraudFeatureEngineer

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# Constants
# -----------------------------
EXPECTED_FEATURES: List[str] = [
    "hour_of_day",
    "day_of_week",
    "time_since_signup",
    "transactions_per_user",
    "transactions_last_24h"
]

# -----------------------------
# Dataclasses
# -----------------------------
@dataclass
class SampleFraudData:
    user_id: List[int]
    signup_time: List[str]
    purchase_time: List[str]
    purchase_value: List[int]

# -----------------------------
# Utility Functions
# -----------------------------
def create_sample_fraud_df() -> pd.DataFrame:
    """
    Creates a sample DataFrame for FraudFeatureEngineer tests.
    
    Returns:
        pd.DataFrame: Sample transactions data.
    """
    logger.info("Creating sample fraud DataFrame for feature engineering")
    data = SampleFraudData(
        user_id=[1, 1, 2, 2],
        signup_time=[
            "2025-01-01 08:00:00", "2025-01-01 08:00:00",
            "2025-01-02 09:00:00", "2025-01-02 09:00:00"
        ],
        purchase_time=[
            "2025-01-01 10:00:00", "2025-01-01 15:00:00",
            "2025-01-02 10:00:00", "2025-01-03 11:00:00"
        ],
        purchase_value=[100, 150, 200, 250]
    )
    df = pd.DataFrame(data.__dict__)
    logger.debug("Sample fraud DataFrame created:\n%s", df.head())
    return df

# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def sample_fraud_df() -> pd.DataFrame:
    """
    Pytest fixture providing a sample fraud DataFrame.
    """
    return create_sample_fraud_df()

# -----------------------------
# Test Function
# -----------------------------
def test_features_created(sample_fraud_df: pd.DataFrame) -> None:
    """
    Test that FraudFeatureEngineer correctly creates all expected features.
    
    Args:
        sample_fraud_df (pd.DataFrame): Sample transactions dataset.
    """
    logger.info("Testing feature creation in FraudFeatureEngineer")
    fe = FraudFeatureEngineer(sample_fraud_df)
    df_feat = (
        fe.parse_timestamps()
          .add_time_features()
          .add_time_since_signup()
          .add_transaction_velocity()
          .get_features()
    )

    for feature in EXPECTED_FEATURES:
        assert feature in df_feat.columns, f"Feature '{feature}' is missing"
    logger.info("All expected features created successfully")
