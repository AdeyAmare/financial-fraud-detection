from __future__ import annotations

# ==========================================================
# IMPORTS
# ==========================================================

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal

import pandas as pd
from pandas.api.types import is_numeric_dtype
from numpy.typing import NDArray

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE


# ==========================================================
# CONSTANTS (NO MAGIC NUMBERS)
# ==========================================================

DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_IMBALANCE_STRATEGY: Literal["SMOTE", None] = "SMOTE"
MIN_ROWS_REQUIRED: int = 10


# ==========================================================
# CUSTOM EXCEPTIONS
# ==========================================================

class DataValidationError(Exception):
    """Raised when schema validation fails."""


class TransformerStateError(Exception):
    """Raised when methods are called in invalid order."""


# ==========================================================
# LOGGING UTILITY
# ==========================================================

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger(__name__)


# ==========================================================
# SCHEMA VALIDATION UTILITIES
# ==========================================================

def validate_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected pandas DataFrame, got {type(df)}")
    if df.empty:
        raise DataValidationError("DataFrame is empty.")
    if len(df) < MIN_ROWS_REQUIRED:
        raise DataValidationError(
            f"DataFrame must contain at least {MIN_ROWS_REQUIRED} rows."
        )


def validate_columns_exist(df: pd.DataFrame, columns: List[str]) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing columns in DataFrame: {missing}")


def validate_numeric_columns(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    invalid = [col for col in numeric_cols if not is_numeric_dtype(df[col])]
    if invalid:
        raise DataValidationError(
            f"Expected numeric dtype for columns: {invalid}"
        )


def validate_no_nulls(df: pd.DataFrame) -> None:
    if df.isnull().values.any():
        raise DataValidationError("Data contains null values.")


# ==========================================================
# CONFIG DATACLASS
# ==========================================================

@dataclass
class TransformerConfig:
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = DEFAULT_RANDOM_STATE
    imbalance_strategy: Literal["SMOTE", None] = DEFAULT_IMBALANCE_STRATEGY


# ==========================================================
# MAIN CLASS
# ==========================================================

class FraudDataTransformer:
    """
    Transform numerical and categorical features for Fraud_Data.csv.

    Features:
    - Standard scaling for numeric columns
    - One-hot encoding for categorical columns
    - Train/test split
    - Optional SMOTE for class imbalance handling
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        numeric_features: List[str],
        categorical_features: List[str],
        test_size: float = DEFAULT_TEST_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE
    ):
        """
        Initialize the FraudDataTransformer.

        Args:
            df (pd.DataFrame): Input dataset.
            target (str): Name of the target column.
            numeric_features (List[str]): Names of numeric feature columns.
            categorical_features (List[str]): Names of categorical feature columns.
            test_size (float, optional): Proportion of data for testing. Defaults to 0.2.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.

        Raises:
            TypeError: If df is not a pandas DataFrame.
            ValueError: If target or features are missing in the DataFrame.
        """

        validate_dataframe(df)

        if target not in df.columns:
            raise DataValidationError(f"Target column '{target}' not found.")

        validate_columns_exist(df, numeric_features + categorical_features)
        validate_numeric_columns(df, numeric_features)
        validate_no_nulls(df)

        self.df: pd.DataFrame = df.copy()
        self.target: str = target
        self.numeric_features: List[str] = numeric_features
        self.categorical_features: List[str] = categorical_features

        self.config: TransformerConfig = TransformerConfig(
            test_size=test_size,
            random_state=random_state
        )

        self.preprocessor: Optional[ColumnTransformer] = None
        self.X_train: Optional[pd.DataFrame | NDArray] = None
        self.X_test: Optional[pd.DataFrame | NDArray] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        self._is_split: bool = False
        self._is_transformed: bool = False

        logger.info("Initialized FraudDataTransformer with %d rows", len(self.df))


    def split_data(self) -> "FraudDataTransformer":
        """
        Split the dataset into training and testing sets.

        Returns:
            FraudDataTransformer: self for method chaining.
        """

        logger.info("Splitting data into train/test sets")

        X = self.df[self.numeric_features + self.categorical_features]
        y = self.df[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )

        self._is_split = True

        logger.info("Train size: %d, Test size: %d",
                    len(self.X_train), len(self.X_test))

        return self


    def transform_features(self) -> "FraudDataTransformer":
        """
        Apply transformations:
        - StandardScaler for numeric columns
        - OneHotEncoder for categorical columns

        Returns:
            FraudDataTransformer: self for method chaining.
        """

        if not self._is_split:
            raise TransformerStateError(
                "split_data() must be called before transform_features()."
            )

        logger.info("Transforming features")

        transformers = []

        if self.numeric_features:
            transformers.append(
                ("num", StandardScaler(), self.numeric_features)
            )

        if self.categorical_features:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(
                        sparse_output=False,
                        handle_unknown="ignore"
                    ),
                    self.categorical_features
                )
            )

        self.preprocessor = ColumnTransformer(transformers)

        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)

        self._is_transformed = True

        logger.info("Features transformed successfully")

        return self


    def handle_imbalance(self, strategy: Optional[str] = DEFAULT_IMBALANCE_STRATEGY) -> "FraudDataTransformer":
        """
        Handle class imbalance in the training set.

        Args:
            strategy (str): Resampling strategy. Currently only 'SMOTE' is supported.

        Returns:
            FraudDataTransformer: self for method chaining.

        Raises:
            ValueError: If unknown strategy is provided.
        """

        if not self._is_transformed:
            raise TransformerStateError(
                "transform_features() must be called before handle_imbalance()."
            )

        if strategy == "SMOTE":
            logger.info("Applying SMOTE to training data")
            smote = SMOTE(random_state=self.config.random_state)
            self.X_train, self.y_train = smote.fit_resample(
                self.X_train, self.y_train
            )
        elif strategy is None:
            logger.warning("No resampling applied.")
        else:
            raise DataValidationError(f"Unknown imbalance strategy: {strategy}")

        return self

    def get_train_test(self) -> Tuple[
        pd.DataFrame | NDArray,
        pd.DataFrame | NDArray,
        pd.Series,
        pd.Series
    ]:
        """
        Retrieve the transformed train/test datasets.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
        """

        if not self._is_transformed:
            raise TransformerStateError(
                "Data has not been fully prepared. "
                "Call split_data() and transform_features() first."
            )

        return self.X_train, self.X_test, self.y_train, self.y_test
