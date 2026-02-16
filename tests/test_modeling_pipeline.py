import pytest
import pandas as pd
import numpy as np
import logging
import warnings
from dataclasses import dataclass
from typing import List

from src.modeling import ModelingPipeline, PipelineConfig

# -----------------------------
# Warnings & Logging Setup
# -----------------------------
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# Constants
# -----------------------------
NUMERIC_FEATURES: List[str] = ["num1", "num2"]
CATEGORICAL_FEATURES: List[str] = ["cat1"]
TARGET_COL: str = "class"
SAMPLE_SIZE: int = 100

# -----------------------------
# Dataclasses
# -----------------------------
@dataclass
class SampleModelingData:
    num1: np.ndarray
    num2: np.ndarray
    cat1: np.ndarray
    clazz: np.ndarray  # renamed from "class" to avoid Python keyword conflict

# -----------------------------
# Utility Functions
# -----------------------------
def create_sample_modeling_df(sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """
    Create a synthetic dataset for testing the modeling pipeline.
    
    Args:
        sample_size (int): Number of rows in the dataset.
    
    Returns:
        pd.DataFrame: Synthetic dataset with numeric, categorical, and target columns.
    """
    logger.info("Creating synthetic dataset for ModelingPipeline test")
    data = SampleModelingData(
        num1=np.random.randn(sample_size),
        num2=np.random.randn(sample_size),
        cat1=np.random.choice(["A", "B"], size=sample_size),
        clazz=np.random.choice([0, 1], size=sample_size)
    )
    df = pd.DataFrame({
        "num1": data.num1,
        "num2": data.num2,
        "cat1": data.cat1,
        "class": data.clazz
    })
    logger.debug("Sample modeling df head:\n%s", df.head())
    return df

# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def sample_data() -> pd.DataFrame:
    """
    Pytest fixture for synthetic modeling data.
    """
    return create_sample_modeling_df()

# -----------------------------
# Test Function
# -----------------------------
def test_pipeline_runs(sample_data: pd.DataFrame) -> None:
    """
    Tests that the ModelingPipeline runs end-to-end including:
        - Data preparation
        - Logistic Regression training
        - Random Forest training
        - Model comparison
        - Best model selection
    
    Args:
        sample_data (pd.DataFrame): Synthetic dataset fixture.
    """
    logger.info("Initializing pipeline configuration and ModelingPipeline")
    config = PipelineConfig(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        target_col=TARGET_COL,
        use_smote=True
    )

    pipeline = ModelingPipeline(df=sample_data, config=config)

    # -----------------------------
    # Test Data Preparation
    # -----------------------------
    X_train, X_test, y_train, y_test = pipeline.prepare_data()
    assert X_train.shape[0] > 0, "X_train is empty"
    assert X_test.shape[0] > 0, "X_test is empty"

    # -----------------------------
    # Test Logistic Regression
    # -----------------------------
    lr_model = pipeline.tune_and_train_logistic_regression()
    assert lr_model is not None, "Logistic Regression model training failed"
    assert len(pipeline.results) > 0, "Results not recorded after LR training"

    # -----------------------------
    # Test Random Forest
    # -----------------------------
    rf_model = pipeline.train_random_forest()
    assert rf_model is not None, "Random Forest model training failed"
    assert len(pipeline.results) > 1, "Results not recorded after RF training"

    # -----------------------------
    # Test Model Comparison
    # -----------------------------
    comparison_df = pipeline.compare_models()
    assert not comparison_df.empty, "Model comparison returned empty DataFrame"

    # -----------------------------
    # Test Selecting Best Model
    # -----------------------------
    best_model, justification = pipeline.select_best_model()
    assert "Model" in best_model, "Best model dict missing 'Model' key"
    assert isinstance(justification, str), "Justification should be a string"

    logger.info("ModelingPipeline end-to-end test passed successfully")
