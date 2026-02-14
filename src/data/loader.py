import pandas as pd
from pathlib import Path
from typing import Any, Set
from dataclasses import dataclass
import logging


# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

LOGGER = logging.getLogger("DataLoader")


# --------------------------------------------------
# Constants (No Magic Values)
# --------------------------------------------------

BINARY_TARGET_VALUES: Set[int] = {0, 1}


# --------------------------------------------------
# Configuration Dataclasses
# --------------------------------------------------

@dataclass(frozen=True)
class FraudLoaderConfig:
    required_columns: Set[str] = frozenset({"signup_time", "purchase_time"})
    target_column: str = "class"


@dataclass(frozen=True)
class CreditCardLoaderConfig:
    required_columns: Set[str] = frozenset({"Time", "Amount", "Class"})
    target_column: str = "Class"


@dataclass(frozen=True)
class IPCountryLoaderConfig:
    required_columns: Set[str] = frozenset({
        "lower_bound_ip_address",
        "upper_bound_ip_address",
        "country",
    })


# --------------------------------------------------
# Shared Utility Functions
# --------------------------------------------------

def _read_csv_file(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df: pd.DataFrame = pd.read_csv(file_path)
        LOGGER.info(f"Loaded file: {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV file: {e}")


def _validate_required_columns(df: pd.DataFrame, required: Set[str]) -> None:
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def _validate_binary_target(df: pd.DataFrame, column: str) -> None:
    if not pd.api.types.is_numeric_dtype(df[column]):
        df[column] = pd.to_numeric(df[column], errors="raise")

    unique_vals = set(df[column].dropna().unique())
    if not unique_vals.issubset(BINARY_TARGET_VALUES):
        raise ValueError(
            f"Target column must contain only {BINARY_TARGET_VALUES}. Found: {unique_vals}"
        )


def _validate_no_negative_values(df: pd.DataFrame, columns: Set[str]) -> None:
    for col in columns:
        if col in df.columns:
            if (df[col] < 0).any():
                raise ValueError(f"Column '{col}' contains negative values")


# --------------------------------------------------
# FraudDataLoader
# --------------------------------------------------

class FraudDataLoader:
    """
    Loader for fraud transaction data CSV files.

    Ensures the presence of required columns and parses datetime columns.

    Required Columns:
        - signup_time
        - purchase_time
    """

    REQUIRED_COLUMNS = {"signup_time", "purchase_time"}
    TARGET_COLUMN = "class"

    def __init__(self, file_path: str):
        self.file_path: Path = Path(file_path)
        self.config = FraudLoaderConfig()

    def load(self) -> pd.DataFrame:
        df: pd.DataFrame = _read_csv_file(self.file_path)

        _validate_required_columns(df, self.config.required_columns)

        if self.config.target_column in df.columns:
            _validate_binary_target(df, self.config.target_column)

        try:
            df["signup_time"] = pd.to_datetime(df["signup_time"], errors="raise")
            df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="raise")
        except Exception as e:
            raise ValueError(f"Datetime parsing failed: {e}")

        if (df["purchase_time"] < df["signup_time"]).any():
            raise ValueError("Found purchase_time earlier than signup_time")

        LOGGER.info("Fraud dataset validation complete")

        return df


# --------------------------------------------------
# CreditCardDataLoader
# --------------------------------------------------

class CreditCardDataLoader:
    """
    Loader for credit card data CSV files.
    """

    REQUIRED_COLUMNS = {"Time", "Amount", "Class"}
    TARGET_COLUMN = "Class"

    def __init__(self, file_path: str):
        self.file_path: Path = Path(file_path)
        self.config = CreditCardLoaderConfig()

    def load(self) -> pd.DataFrame:
        df: pd.DataFrame = _read_csv_file(self.file_path)

        _validate_required_columns(df, self.config.required_columns)

        try:
            df["Amount"] = pd.to_numeric(df["Amount"], errors="raise")
            df["Time"] = pd.to_numeric(df["Time"], errors="raise")
        except Exception as e:
            raise ValueError(f"Numeric conversion failed: {e}")

        _validate_no_negative_values(df, {"Amount", "Time"})

        _validate_binary_target(df, self.config.target_column)

        LOGGER.info("Credit card dataset validation complete")

        return df


# --------------------------------------------------
# IPCountryLoader
# --------------------------------------------------

class IPCountryLoader:
    """
    Loader for IP-to-country mapping CSV files.

    Ensures the presence of required columns:
        - lower_bound_ip_address
        - upper_bound_ip_address
        - country
    """

    REQUIRED_COLUMNS = {
        "lower_bound_ip_address",
        "upper_bound_ip_address",
        "country",
    }

    def __init__(self, file_path: str):
        self.file_path: Path = Path(file_path)
        self.config = IPCountryLoaderConfig()

    def load(self) -> pd.DataFrame:
        df: pd.DataFrame = _read_csv_file(self.file_path)

        _validate_required_columns(df, self.config.required_columns)

        try:
            df["lower_bound_ip_address"] = pd.to_numeric(
                df["lower_bound_ip_address"], errors="raise"
            )
            df["upper_bound_ip_address"] = pd.to_numeric(
                df["upper_bound_ip_address"], errors="raise"
            )
        except Exception as e:
            raise ValueError(f"IP bounds conversion failed: {e}")

        if (df["lower_bound_ip_address"] > df["upper_bound_ip_address"]).any():
            raise ValueError(
                "Found lower_bound_ip_address greater than upper_bound_ip_address"
            )

        LOGGER.info("IP country dataset validation complete")

        return df
