from typing import Union, Set
import pandas as pd
import numpy as np

import logging
from dataclasses import dataclass

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
LOGGER = logging.getLogger("GeoDataMerger")


# -----------------------------
# Dataclasses for configuration
# -----------------------------
@dataclass(frozen=True)
class IPConfig:
    lower_col: str = "lower_bound_ip_address"
    upper_col: str = "upper_bound_ip_address"
    country_col: str = "country"
    unknown_label: str = "Unknown"


@dataclass(frozen=True)
class GeoMergerConfig:
    ip_invalid_value: int = -1
    default_sort_col: str = "ip_int"

# -----------------------------
# Utility functions
# -----------------------------
def ip_to_int(ip: Union[str, float, int], invalid_value: int = -1) -> int:
    """
    Convert IP address to integer, return `invalid_value` if invalid/missing.
    """
    if isinstance(ip, (int, float)):
        if pd.isna(ip):
            return invalid_value
        return int(ip)

    if isinstance(ip, str):
        try:
            parts = ip.strip().split(".")
            if len(parts) != 4:
                return invalid_value
            value = 0
            for part in parts:
                value = value * 256 + int(part)
            return value
        except (ValueError, AttributeError):
            return invalid_value

    return invalid_value


def validate_ip_range(df: pd.DataFrame, lower_col: str, upper_col: str) -> None:
    """
    Ensure all lower bounds <= upper bounds, raise ValueError if violated.
    """
    if (df[lower_col] > df[upper_col]).any():
        raise ValueError("Found lower_bound_ip_address greater than upper_bound_ip_address")


# -----------------------------
# GeoDataMerger Class
# -----------------------------

class GeoDataMerger:
    """
    Class-based wrapper for IP-to-country geolocation utilities.

    Features:
    - Convert IP addresses to integers
    - Prepare IP range data for efficient merging
    - Merge transaction data with country information based on IP
    - Generate country-level fraud statistics

    Attributes:
        transactions_df (pd.DataFrame): Transaction dataset with IPs.
        ip_df (pd.DataFrame): IP-to-country mapping dataset.
        ip_column (str): Column name in transactions_df containing IP addresses.
        target_col (str): Column in transactions_df indicating fraud (1) / non-fraud (0).
    """

    IP_CONFIG: IPConfig = IPConfig()
    CONFIG: GeoMergerConfig = GeoMergerConfig()

    def __init__(
        self,
        transactions_df: pd.DataFrame,
        ip_df: pd.DataFrame,
        ip_column: str = "ip_address",
        target_col: str = "class",
    ):
        """
        Initialize the GeoDataMerger.

        Args:
            transactions_df (pd.DataFrame): Transactions dataset.
            ip_df (pd.DataFrame): IP-to-country mapping dataset.
            ip_column (str): Column in transactions_df containing IPs.
            target_col (str): Column indicating fraud class.

        Raises:
            TypeError: If inputs are not pandas DataFrames.
        """
        if not isinstance(transactions_df, pd.DataFrame):
            raise TypeError(f"transactions_df must be a pandas DataFrame, got {type(transactions_df)}")
        if not isinstance(ip_df, pd.DataFrame):
            raise TypeError(f"ip_df must be a pandas DataFrame, got {type(ip_df)}")
        
        if transactions_df.empty:
            raise ValueError("transactions_df is empty")
        if ip_df.empty:
            raise ValueError("ip_df is empty")

        self.transactions_df: pd.DataFrame = transactions_df.copy()
        self.ip_df: pd.DataFrame = ip_df.copy()
        self.ip_column: str = ip_column
        self.target_col: str = target_col
        self.logger: logging.Logger = LOGGER

    

    def ip_series_to_int(self, ip_series: pd.Series) -> pd.Series:
        """
        Convert a pandas Series of IPs to integer representation.

        Args:
            ip_series (pd.Series): Series of IP addresses.

        Returns:
            pd.Series: Integer representation of IPs.
        """
        if pd.api.types.is_numeric_dtype(ip_series):
            return ip_series.fillna(self.CONFIG.ip_invalid_value).astype(np.int64)
        return ip_series.apply(lambda x: ip_to_int(x, invalid_value=self.CONFIG.ip_invalid_value))


    # -----------------------------
    # IP range preparation
    # -----------------------------
    def prepare_ip_ranges(self) -> pd.DataFrame:
        """
        Prepare the IP-to-country mapping DataFrame:
        - Convert bounds to integers
        - Fill missing values
        - Sort by lower_bound_ip_address

        Returns:
            pd.DataFrame: Prepared IP range DataFrame.
        """
        df = self.ip_df.copy()
        df[self.IP_CONFIG.lower_col] = pd.to_numeric(df[self.IP_CONFIG.lower_col], errors="coerce").fillna(0).astype(np.int64)
        df[self.IP_CONFIG.upper_col] = pd.to_numeric(df[self.IP_CONFIG.upper_col], errors="coerce").fillna(0).astype(np.int64)
        validate_ip_range(df, self.IP_CONFIG.lower_col, self.IP_CONFIG.upper_col)
        df = df.sort_values(self.IP_CONFIG.lower_col).reset_index(drop=True)
        return df

    # -----------------------------
    # Merge logic (merge_asof)
    # -----------------------------
    def merge_country(self) -> pd.DataFrame:
        """
        Merge transaction data with country information based on IP.

        Handles:
        - Invalid IPs
        - IPs outside defined ranges

        Returns:
            pd.DataFrame: Transactions dataset with a new 'country' column.
        """
        df = self.transactions_df.copy()

        # Convert IPs to integers
        df["ip_int"] = self.ip_series_to_int(df[self.ip_column])

        # Prepare IP ranges
        ip_ranges = self.prepare_ip_ranges()

        # Sort for merge_asof
        df_sorted = df.sort_values("ip_int").reset_index()

        merged = pd.merge_asof(
            df_sorted,
            ip_ranges[[self.IP_CONFIG.lower_col, self.IP_CONFIG.upper_col, self.IP_CONFIG.country_col]],
            left_on="ip_int",
            right_on=self.IP_CONFIG.lower_col,
            direction="backward"
        )

        # Validate upper bound + invalid IPs
        invalid_mask = (
            (merged["ip_int"] > merged[self.IP_CONFIG.upper_col])
            | (merged[self.IP_CONFIG.lower_col].isna())
            | (merged["ip_int"] < 0)
        )
        merged.loc[invalid_mask, self.IP_CONFIG.country_col] = self.IP_CONFIG.unknown_label

        # Cleanup temporary columns
        merged = (
            merged.sort_values("index")
            .drop(columns=["index", "ip_int", self.IP_CONFIG.lower_col, self.IP_CONFIG.upper_col])
            .reset_index(drop=True)
        )

        self.transactions_df = merged
        return self.transactions_df

    # -----------------------------
    # Country fraud statistics
    # -----------------------------
    def get_summary(self) -> pd.DataFrame:
        """
        Generate country-level fraud statistics:
        - Total transactions per country
        - Fraud count per country
        - Fraud rate per country

        Returns:
            pd.DataFrame: Aggregated country statistics sorted by fraud_count.
        """
        stats = self.transactions_df.groupby(self.IP_CONFIG.country_col).agg(
            total_transactions=(self.target_col, "count"),
            fraud_count=(self.target_col, "sum")
        ).reset_index()

        stats["fraud_rate"] = stats["fraud_count"] / stats["total_transactions"]
        stats = stats.sort_values("fraud_count", ascending=False).reset_index(drop=True)
        return stats
