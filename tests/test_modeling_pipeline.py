import pytest
import pandas as pd
import numpy as np
from src.modeling import ModelingPipeline  
import warnings
warnings.filterwarnings("ignore")


# Minimal synthetic dataset
@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        "num1": np.random.randn(100),
        "num2": np.random.randn(100),
        "cat1": np.random.choice(["A", "B"], size=100),
        "class": np.random.choice([0, 1], size=100)
    })
    return data

def test_pipeline_runs(sample_data):
    pipeline = ModelingPipeline(
        df=sample_data,
        numeric_features=["num1", "num2"],
        categorical_features=["cat1"],
        target_col="class",
        use_smote=True
    )
    
    # Test data preparation
    X_train, X_test, y_train, y_test = pipeline.prepare_data()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    
    # Test training Logistic Regression
    lr_model = pipeline.train_logistic_regression()
    assert lr_model is not None
    assert len(pipeline.results) > 0

    # Test training Random Forest
    rf_model = pipeline.train_random_forest()
    assert rf_model is not None
    assert len(pipeline.results) > 1

    # Test model comparison
    comparison_df = pipeline.compare_models()
    assert not comparison_df.empty

    # Test selecting best model
    best_model, justification = pipeline.select_best_model()
    assert "Model" in best_model
    assert isinstance(justification, str)
