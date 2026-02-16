import joblib
import logging
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ----------------------------
# Constants
# ----------------------------
DEFAULT_TOP_N = 10
DEFAULT_SAMPLE_SIZE = 500
SHAP_BACKGROUND_SIZE = 100
RANDOM_STATE = 42
VALID_CASE_TYPES = {"TP", "FP", "FN"}

# Ensure JavaScript is initialized for SHAP plots in Notebooks
try:
    import IPython
    shap.initjs()
except ImportError:
    pass  # Skip initjs outside Jupyter/notebook


# ----------------------------
# Centralized logging
# ----------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------
# Data classes for configuration
# ----------------------------
@dataclass(frozen=True)
class ExplainabilityConfig:
    top_n: int = DEFAULT_TOP_N
    sample_size: int = DEFAULT_SAMPLE_SIZE
    shap_background_size: int = SHAP_BACKGROUND_SIZE
    random_state: int = RANDOM_STATE

# ----------------------------
# Utility functions
# ----------------------------
def sample_array(X: np.ndarray, sample_size: int, random_state: int) -> np.ndarray:
    if X.shape[0] > sample_size:
        return shap.sample(X, sample_size, random_state=random_state)
    return X

def flatten_instance(X_instance: Union[np.ndarray, object]) -> np.ndarray:
    if hasattr(X_instance, "toarray"):
        return X_instance.toarray().flatten()
    return np.array(X_instance).flatten()

def select_indices_by_case(y_true: np.ndarray, y_pred: np.ndarray, case_type: str) -> np.ndarray:
    if case_type == "TP":
        return np.where((y_true == 1) & (y_pred == 1))[0]
    if case_type == "FP":
        return np.where((y_true == 0) & (y_pred == 1))[0]
    if case_type == "FN":
        return np.where((y_true == 1) & (y_pred == 0))[0]
    return np.array([])

def extract_shap_array(shap_values: np.ndarray) -> np.ndarray:
    if hasattr(shap_values, "values"):
        arr = shap_values.values
    else:
        arr = shap_values
    if len(arr.shape) == 3:  # multiclass SHAP
        arr = arr[:, :, 1]
    return np.ravel(arr)

# ----------------------------
# Main Explainability Class
# ----------------------------
class ModelExplainability:
    """
    ModelExplainability provides SHAP-based interpretability utilities
    for trained fraud detection models that rely on a saved preprocessing
    pipeline.

    This class supports:
    - Built-in feature importance visualization
    - Global SHAP summary plots
    - Instance-level SHAP force plots (TP / FP / FN)
    - Extraction of top SHAP feature drivers

    Notes
    -----
    - The model is assumed to be already trained.
    - The preprocessor must expose `transform()` and `get_feature_names_out()`.
    - SHAP logic is intentionally defensive to handle tree and non-tree models.
    """

    def __init__(
        self,
        model_path: str,
        preprocessor_path: str,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame],
        config: ExplainabilityConfig = ExplainabilityConfig()
    ):
        """
        Initialize the explainability module.

        Parameters
        ----------
        model_path : str
            Path to the saved trained model (joblib).
        preprocessor_path : str
            Path to the saved preprocessing pipeline (joblib).
        X : pd.DataFrame
            Raw feature dataframe (before preprocessing).
        y : pd.Series
            Ground-truth labels aligned with X.

        Raises
        ------
        TypeError
            If X or y are not pandas objects.
        RuntimeError
            If model or preprocessor loading fails.
        """
        logger.info("Initializing ModelExplainability")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            raise TypeError("y must be a pandas Series or DataFrame")

        try:
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
        except Exception as e:
            logger.error("Failed to load model or preprocessor", exc_info=True)
            raise RuntimeError("Model or preprocessor loading failed") from e

        self.X_raw: pd.DataFrame = X.copy()
        self.y: pd.Series = y.reset_index(drop=True)
        self.config: ExplainabilityConfig = config

        try:
            self.X: np.ndarray = self.preprocessor.transform(X)
            self.feature_names: np.ndarray = self.preprocessor.get_feature_names_out()
        except Exception as e:
            logger.error("Preprocessing failed", exc_info=True)
            raise RuntimeError("Error during feature preprocessing") from e

        try:
            self.predictions: np.ndarray = self.model.predict(self.X)
        except Exception as e:
            logger.error("Prediction failed", exc_info=True)
            raise RuntimeError("Model prediction failed") from e

        self.probabilities: Optional[np.ndarray] = (
            self.model.predict_proba(self.X)[:, 1]
            if hasattr(self.model, "predict_proba")
            else None
        )

        self.explainer: Optional[object] = None
        self.shap_values: Optional[np.ndarray] = None
        self.X_shap: Optional[np.ndarray] = None

        logger.info("ModelExplainability initialized successfully")

    def plot_builtin_feature_importance(self, top_n: int = DEFAULT_TOP_N) -> Optional[pd.DataFrame]:
        """
        Plot built-in feature importances for tree-based models.

        Parameters
        ----------
        top_n : int, default=10
            Number of top features to display.

        Returns
        -------
        pd.DataFrame or None
            DataFrame of top features if supported, otherwise None.
        """
        if not hasattr(self.model, "feature_importances_"):
            logger.warning("Model does not support built-in feature importance")
            return None

        importances = self.model.feature_importances_
        if len(importances) != len(self.feature_names):
            logger.warning("Feature importance and feature name length mismatch")

        importance_df = (
            pd.DataFrame({"feature": self.feature_names, "importance": importances})
            .sort_values(by="importance", ascending=False)
            .head(top_n)
        )

        plt.figure(figsize=(8, 6))
        plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Feature Importances (Built-in)")
        plt.tight_layout()
        plt.show()

        return importance_df

    def compute_shap_values(self, sample_size: int = DEFAULT_SAMPLE_SIZE) -> None:
        """
        Compute SHAP values for the model.

        Parameters
        ----------
        sample_size : int, default=500
            Maximum number of samples used for SHAP computation.
        """
        logger.info("Computing SHAP values")

        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError("sample_size must be a positive integer")

        try:
            self.X_shap = sample_array(self.X, sample_size, self.config.random_state)

            if hasattr(self.model, "estimators_"):
                logger.info("Using TreeExplainer")
                self.explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.explainer.shap_values(self.X_shap)
                if isinstance(self.shap_values, list):
                    self.shap_values = self.shap_values[1]
            else:
                logger.info("Using model-agnostic SHAP Explainer")
                background = shap.sample(self.X, self.config.shap_background_size, random_state=self.config.random_state)
                self.explainer = shap.Explainer(self.model, background)
                self.shap_values = self.explainer(self.X_shap)

        except Exception as e:
            logger.error("SHAP computation failed", exc_info=True)
            raise RuntimeError("Error during SHAP computation") from e

    def plot_shap_summary(self, max_display: int = 20) -> None:
        """
        Plot global SHAP summary plot.

        Parameters
        ----------
        max_display : int, default=20
            Maximum number of features to display.
        """
        if self.shap_values is None:
            self.compute_shap_values()

        shap_vals_to_plot = (
            self.shap_values.values
            if hasattr(self.shap_values, "values")
            else self.shap_values
        )

        shap.summary_plot(
            shap_vals_to_plot,
            features=self.X_shap,
            feature_names=self.feature_names,
            max_display=max_display
        )

    def plot_force_plot_for_case(self, case_type: str = "TP"):
        """
        Generate a SHAP force plot for a specific confusion-matrix case.

        Parameters
        ----------
        case_type : {"TP", "FP", "FN"}
            Type of prediction case.

        Returns
        -------
        shap.plots._force.AdditiveForceVisualizer or None
        """
        logger.info(f"Generating force plot for case: {case_type}")

        if case_type not in VALID_CASE_TYPES:
            raise ValueError(f"case_type must be one of {VALID_CASE_TYPES}")

        y_true, y_pred = self.y.values, self.predictions
        idx_list = select_indices_by_case(y_true, y_pred, case_type)

        if len(idx_list) == 0:
            logger.warning(f"No samples found for {case_type}")
            return None

        idx = idx_list[0]
        X_instance = flatten_instance(self.X[idx])

        try:
            if hasattr(self.model, "estimators_") or "XGB" in str(type(self.model)):
                explainer = shap.TreeExplainer(self.model)
                raw_vals = explainer.shap_values(self.X[idx:idx + 1])

                if isinstance(raw_vals, list):
                    shap_vals = raw_vals[1].flatten()
                    base_val = explainer.expected_value[1]
                elif len(raw_vals.shape) == 3:
                    shap_vals = raw_vals[0, :, 1].flatten()
                    base_val = explainer.expected_value[1]
                else:
                    shap_vals = raw_vals.flatten()
                    base_val = explainer.expected_value
            else:
                background = shap.sample(self.X, self.config.shap_background_size, random_state=self.config.random_state)
                explainer = shap.Explainer(self.model, background)
                shap_exp = explainer(self.X[idx:idx + 1])

                if len(shap_exp.values.shape) == 3:
                    shap_vals = shap_exp.values[0, :, 1].flatten()
                    base_val = shap_exp.base_values[0, 1]
                else:
                    shap_vals = shap_exp.values[0].flatten()
                    base_val = shap_exp.base_values[0]

        except Exception as e:
            logger.error("SHAP force plot computation failed", exc_info=True)
            raise RuntimeError("Force plot SHAP computation failed") from e

        if len(shap_vals) == 2 * len(X_instance):
            logger.warning("Detected interleaved SHAP values, correcting dimensions")
            shap_vals = shap_vals[len(X_instance):]

        return shap.force_plot(
            base_value=float(base_val),
            shap_values=shap_vals,
            features=X_instance,
            feature_names=self.feature_names
        )

    def get_top_drivers(self, top_n: int = 5) -> pd.DataFrame:
        """
        Retrieve top SHAP feature drivers by mean absolute contribution.

        Parameters
        ----------
        top_n : int, default=5
            Number of top drivers to return.

        Returns
        -------
        pd.DataFrame
            Top features ranked by mean absolute SHAP value.
        """
        if self.shap_values is None:
            self.compute_shap_values()

        # Ensure shap_array is 2D (samples x features)
        if hasattr(self.shap_values, "values"):
            shap_array = self.shap_values.values
        else:
            shap_array = self.shap_values

        # If multiclass, take positive class
        if len(shap_array.shape) == 3:
            shap_array = shap_array[:, :, 1]

        # Compute mean absolute shap for each feature
        mean_abs_shap = np.abs(shap_array).mean(axis=0)

        # Ensure feature count matches
        if mean_abs_shap.shape[0] != len(self.feature_names):
            raise ValueError(
                f"Number of SHAP feature values ({mean_abs_shap.shape[0]}) "
                f"does not match number of features ({len(self.feature_names)})."
            )

        return (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs_shap})
            .sort_values(by="mean_abs_shap", ascending=False)
            .head(top_n)
        )
