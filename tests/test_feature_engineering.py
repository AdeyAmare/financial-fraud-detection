import pytest
import pandas as pd
from src.feature_engineering import FraudFeatureEngineer

@pytest.fixture
def sample_fraud_df():
    return pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "signup_time": ["2025-01-01 08:00:00", "2025-01-01 08:00:00",
                        "2025-01-02 09:00:00", "2025-01-02 09:00:00"],
        "purchase_time": ["2025-01-01 10:00:00", "2025-01-01 15:00:00",
                          "2025-01-02 10:00:00", "2025-01-03 11:00:00"],
        "purchase_value": [100, 150, 200, 250]
    })

def test_features_created(sample_fraud_df):
    fe = FraudFeatureEngineer(sample_fraud_df)
    df_feat = (fe
               .parse_timestamps()
               .add_time_features()
               .add_time_since_signup()
               .add_transaction_velocity()
               .get_features())

    # Just check that the new features exist
    expected_features = [
        'hour_of_day', 
        'day_of_week', 
        'time_since_signup', 
        'transactions_per_user', 
        'transactions_last_24h'
    ]

    for feature in expected_features:
        assert feature in df_feat.columns
