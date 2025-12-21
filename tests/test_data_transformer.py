import pytest
import pandas as pd
from src.transformation_and_imbalance import FraudDataTransformer

@pytest.fixture
def sample_df():
    
    data = {
        "purchase_value": [100 + i*10 for i in range(20)],
        "age": [20 + i for i in range(20)],
        "sex": ["M", "F"] * 10,
        "browser": ["Chrome", "Safari", "Firefox", "Edge", "Opera"] * 4,
        "class": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    return pd.DataFrame(data)


def test_fraud_data_transformer_pipeline(sample_df):
    numeric_features = ["purchase_value", "age"]
    categorical_features = ["sex", "browser"]
    target = "class"

    transformer = FraudDataTransformer(
        df=sample_df,
        target=target,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    # Run the full pipeline
    transformer.split_data().transform_features().handle_imbalance()
    X_train, X_test, y_train, y_test = transformer.get_train_test()

    # Basic checks
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[1] > 0  # numeric + encoded categorical features
    assert X_test.shape[1] == X_train.shape[1]
