import pandas as pd
import logging
from typing import Literal, Set
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

LOGGER = logging.getLogger("TransactionDataCleaner")

# -----------------------------
# Dataclasses for configuration
# -----------------------------
@dataclass(frozen=True)
class TargetConfig:
    columns: Set[str] = frozenset({"class", "Class"})
    valid_values: Set[int] = frozenset({0, 1})


@dataclass(frozen=True)
class MissingValueConfig:
    strategies: Set[str] = frozenset({"drop", "fill_zero", "fill_median"})


# -----------------------------
# Utility Functions
# -----------------------------
def _validate_binary_column(df: pd.DataFrame, col: str, valid_values: Set[int]) -> None:
    """
    Ensure a column is numeric and binary (0 or 1).
    Raises ValueError if invalid.
    """
    df[col] = pd.to_numeric(df[col], errors="coerce")
    unique_vals = set(df[col].dropna().unique())
    if not unique_vals.issubset(valid_values):
        raise ValueError(f"{col} must contain only {valid_values}. Found: {unique_vals}")



class TransactionDataCleaner:
    """
    Class for cleaning transaction datasets.

    Supports:
    - Removing duplicate rows
    - Handling missing values with various strategies
    - Correcting data types for specific columns
    - Reporting dataset issues including duplicates, missing values, and data types

    Attributes:
        df (pd.DataFrame): The transaction DataFrame to clean.
        logger (logging.Logger): Logger instance for the class.
    """

    TARGET_CONFIG: TargetConfig = TargetConfig()
    MISSING_CONFIG: MissingValueConfig = MissingValueConfig()

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the TransactionDataCleaner with a DataFrame.

        Args:
            df (pd.DataFrame): Input transaction dataset.

        Raises:
            TypeError: If the provided df is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(df)}")

        if df.empty:  # ADDED
            raise ValueError("Input DataFrame is empty")

        self.df: pd.DataFrame = df.copy()
        self.logger: logging.Logger = LOGGER

    # ---------------------------
    # 1️⃣ Report method
    # ---------------------------
    def report(self) -> None:
        """
        Log comprehensive information about the dataset:
        - Shape
        - Data types per column
        - Number of duplicate rows
        - Missing values count and percentage per column
        """
        try:
            self.logger.info(f"Dataset shape: {self.df.shape}")
            self.logger.info(f"Column data types:\n{self.df.dtypes}")

            duplicate_count = self.df.duplicated().sum()
            self.logger.info(f"Number of duplicate rows: {duplicate_count}")

            missing_per_column = self.df.isna().sum()
            total_missing = missing_per_column.sum()

            # ADDED: Safe division
            if len(self.df) > 0:
                missing_pct = (missing_per_column / len(self.df)) * 100
            else:
                missing_pct = 0

            self.logger.info(f"Missing values per column:\n{missing_per_column}")
            self.logger.info(f"Missing values (%) per column:\n{missing_pct}")
            self.logger.info(f"Total missing values: {total_missing}")

            # ADDED: Log class imbalance if target exists
            for col in self.TARGET_CONFIG.columns:
                if col in self.df.columns:
                    fraud_ratio = self.df[col].mean()
                    self.logger.info(f"{col} fraud ratio: {fraud_ratio:.6f}")

        except Exception as e:
            self.logger.error(f"Error during report generation: {e}")

    # ---------------------------
    # 2️⃣ Remove duplicates
    # ---------------------------
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows from the DataFrame.

        Returns:
            pd.DataFrame: DataFrame with duplicates removed.

        Raises:
            Exception: If removing duplicates fails.
        """
        try:
            initial_rows = len(self.df)
            self.df = self.df.drop_duplicates()

            removed = initial_rows - len(self.df)
            self.logger.info(f"Removed {removed} duplicate rows")

            # ADDED: Ensure dataset not wiped out
            if self.df.empty:
                raise ValueError("All rows removed after duplicate removal")

            return self.df
        except Exception as e:
            self.logger.error(f"Failed to remove duplicates: {e}")
            raise

    # ---------------------------
    # 3️⃣ Handle missing values
    # ---------------------------
    def handle_missing_values(
        self,
        strategy: Literal["drop", "fill_zero", "fill_median"] = "drop"
    ) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.

        Args:
            strategy (str): Strategy to handle missing values:
                - 'drop': Drop rows with any missing values
                - 'fill_zero': Fill missing values with 0
                - 'fill_median': Fill missing numeric columns with median

        Returns:
            pd.DataFrame: DataFrame with missing values handled.

        Raises:
            ValueError: If an unknown strategy is provided.
            Exception: If filling or dropping fails.
        """
        try:
            if strategy not in self.MISSING_CONFIG.strategies:
                raise ValueError(f"Unknown missing value strategy: {strategy}")

            missing_before = self.df.isna().sum().sum()

            if missing_before == 0:
                self.logger.info("No missing values detected")
                return self.df

            if strategy == "drop":
                self.df = self.df.dropna()

                # ADDED: Prevent silent full wipe
                if self.df.empty:
                    raise ValueError("All rows dropped due to missing values")

            elif strategy == "fill_zero":
                self.df = self.df.fillna(0)

            elif strategy == "fill_median":
                numeric_cols = self.df.select_dtypes(include="number").columns
                for col in numeric_cols:
                    median = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median)  # FIXED inplace

            else:
                raise ValueError(f"Unknown missing value strategy: {strategy}")

            missing_after = self.df.isna().sum().sum()
            self.logger.info(
                f"Missing values before: {missing_before}, after: {missing_after}"
            )

            return self.df

        except Exception as e:
            self.logger.error(f"Error handling missing values: {e}")
            raise

    # ---------------------------
    # 4️⃣ Correct data types
    # ---------------------------
    def correct_dtypes(self) -> pd.DataFrame:
        """
        Correct data types for specific columns.

        Currently implemented:
        - Convert 'age' column to integer if exists

        Returns:
            pd.DataFrame: DataFrame with corrected data types.
        """
        try:
            if "age" in self.df.columns:
                self.df["age"] = (
                    pd.to_numeric(self.df["age"], errors="coerce")
                    .astype("Int64")
                )

                # ADDED: basic sanity check
                if (self.df["age"] < 0).any():
                    self.logger.warning("Negative age values detected")

            # ADDED: Ensure target columns are numeric binary
            for col in self.TARGET_CONFIG.columns:
                if col in self.df.columns:
                    _validate_binary_column(self.df, col, self.TARGET_CONFIG.valid_values)

        except Exception as e:
            self.logger.warning(f"Could not convert dtypes properly: {e}")

        return self.df

    # ---------------------------
    # 5️⃣ Full cleaning pipeline
    # ---------------------------
    def clean(
        self,
        missing_strategy: Literal["drop", "fill_zero", "fill_median"] = "drop"
    ) -> pd.DataFrame:
        """
        Run the full data cleaning pipeline:
        1. Remove duplicates
        2. Handle missing values
        3. Correct data types

        Args:
            missing_strategy (str): Strategy for handling missing values.

        Returns:
            pd.DataFrame: Cleaned DataFrame.

        Raises:
            Exception: If any step of the cleaning pipeline fails.
        """
        try:
            self.logger.info(f"Starting cleaning with strategy: {missing_strategy}")  # ADDED

            self.remove_duplicates()
            self.handle_missing_values(strategy=missing_strategy)
            self.correct_dtypes()

            if self.df.empty:  # ADDED
                raise ValueError("Cleaning resulted in empty DataFrame")

            return self.df

        except Exception as e:
            self.logger.error(f"Data cleaning failed: {e}")
            raise
