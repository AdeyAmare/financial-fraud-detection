from asyncio.log import logger
import pandas as pd
import logging

from dataclasses import dataclass
from typing import List, Literal, Optional

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
LOGGER = logging.getLogger("FeatureEngineer")

# -----------------------------
# Configuration Dataclass
# -----------------------------
@dataclass
class FraudFeatureConfig:
    """
    Configuration for FraudFeatureEngineer.
    """
    time_since_signup_unit: Literal["seconds", "minutes", "hours", "days"] = "hours"
    rolling_window: str = "24H"  # pandas-compatible window string
    user_col: str = "user_id"
    signup_col: str = "signup_time"
    purchase_col: str = "purchase_time"


# -----------------------------
# Utility Functions
# -----------------------------
def validate_columns_exist(df: pd.DataFrame, columns: List[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def convert_to_datetime(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        df[col] = pd.to_datetime(df[col], errors="raise")
    return df


def compute_time_difference(
    df: pd.DataFrame,
    start_col: str,
    end_col: str,
    new_col: str,
    unit: Literal["seconds", "minutes", "hours", "days"] = "hours",
) -> pd.DataFrame:
    delta = df[end_col] - df[start_col]
    factor = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}[unit]
    df[new_col] = delta.dt.total_seconds() / factor
    return df


def compute_transactions_per_user(df: pd.DataFrame, user_col: str) -> pd.DataFrame:
    df["transactions_per_user"] = df.groupby(user_col)[user_col].transform("count")
    return df


def compute_rolling_transactions(
    df: pd.DataFrame,
    user_col: str,
    time_col: str,
    rolling_window: str,
    new_col: str,
) -> pd.DataFrame:
    df = df.sort_values([user_col, time_col]).copy()
    df["_txn"] = 1
    df = df.set_index(time_col)
    df[new_col] = (
        df.groupby(user_col)["_txn"]
        .rolling(rolling_window)
        .count()
        .reset_index(level=0, drop=True)
    )
    df = df.reset_index().drop(columns=["_txn"])
    return df

# -----------------------------
# Fraud Feature Engineer Class
# -----------------------------

class FraudFeatureEngineer:
    """
    Feature engineering pipeline for Fraud_Data.csv.

    Provides methods to:
    - Parse timestamp columns
    - Add time-based features (hour of day, day of week)
    - Calculate time since signup
    - Compute transaction velocity features (total transactions per user, rolling 24h count)
    """

    REQUIRED_COLUMNS = ["signup_time", "purchase_time", "user_id"]

    def __init__(self, df: pd.DataFrame, config: FraudFeatureConfig = FraudFeatureConfig()):
        """
        Initialize the FraudFeatureEngineer.

        Args:
            df (pd.DataFrame): Input fraud transaction dataset.

        Raises:
            TypeError: If df is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(df)}")
        self.df: pd.DataFrame = df.copy()
        self.config: FraudFeatureConfig = config

        self.logger: logging.Logger = LOGGER

        self.logger.info("Initialized FraudFeatureEngineer with %d rows", len(self.df))

    def parse_timestamps(self) -> "FraudFeatureEngineer":
        """
        Convert 'signup_time' and 'purchase_time' columns to datetime.

        Returns:
            FraudFeatureEngineer: self for method chaining.

        Raises:
            KeyError: If required columns are missing.
            ValueError: If datetime conversion fails.
        """
        validate_columns_exist(self.df, [self.config.signup_col, self.config.purchase_col])
        self.df = convert_to_datetime(self.df, [self.config.signup_col, self.config.purchase_col])
        self.logger.info("Parsed timestamps: %s, %s", self.config.signup_col, self.config.purchase_col)
        return self

    def add_time_features(self) -> "FraudFeatureEngineer":
        """
        Add time-based features:
        - hour_of_day: hour of the purchase
        - day_of_week: day of the week (0=Monday, 6=Sunday)

        Returns:
            FraudFeatureEngineer: self for method chaining.

        Raises:
            KeyError: If 'purchase_time' column is missing.
        """
        validate_columns_exist(self.df, [self.config.purchase_col])
        self.df["hour_of_day"] = self.df[self.config.purchase_col].dt.hour
        self.df["day_of_week"] = self.df[self.config.purchase_col].dt.dayofweek
        self.logger.info("Added hour_of_day and day_of_week features")
        return self

    def add_time_since_signup(self) -> "FraudFeatureEngineer":
        """
        Add 'time_since_signup' feature in hours.

        Returns:
            FraudFeatureEngineer: self for method chaining.

        Raises:
            KeyError: If 'signup_time' or 'purchase_time' is missing.
        """
        validate_columns_exist(self.df, [self.config.signup_col, self.config.purchase_col])
        self.df = compute_time_difference(
            self.df,
            start_col=self.config.signup_col,
            end_col=self.config.purchase_col,
            new_col="time_since_signup",
            unit=self.config.time_since_signup_unit,
        )
        self.logger.info("Added time_since_signup feature (%s)", self.config.time_since_signup_unit)
        return self

    def add_transaction_velocity(self) -> "FraudFeatureEngineer":
        """
        Add transaction velocity features:
        - transactions_per_user: total transactions per user
        - transactions_last_24h: rolling 24-hour transaction count per user

        Returns:
            FraudFeatureEngineer: self for method chaining.

        Raises:
            KeyError: If 'user_id' or 'purchase_time' columns are missing.
        """
        validate_columns_exist(self.df, [self.config.user_col, self.config.purchase_col])
        self.df = compute_transactions_per_user(self.df, self.config.user_col)
        self.df = compute_rolling_transactions(
            self.df,
            user_col=self.config.user_col,
            time_col=self.config.purchase_col,
            rolling_window=self.config.rolling_window,
            new_col="transactions_last_24h",
        )
        logger.info(
            "Added transactions_per_user and transactions_last_24h features (window=%s)",
            self.config.rolling_window,
        )
        return self

    def get_features(self) -> pd.DataFrame:
        """
        Return the DataFrame with all engineered features.

        Returns:
            pd.DataFrame: Feature-engineered dataset.
        """
        logger.info("Returning engineered feature dataframe")
        return self.df
