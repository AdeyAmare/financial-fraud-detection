from typing import Union
import pandas as pd
import numpy as np


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

        self.transactions_df: pd.DataFrame = transactions_df.copy()
        self.ip_df: pd.DataFrame = ip_df.copy()
        self.ip_column: str = ip_column
        self.target_col: str = target_col

    # -----------------------------
    # IP conversion utilities
    # -----------------------------
    @staticmethod
    def ip_to_int(ip: Union[str, float, int]) -> int:
        """
        Convert an IP address string to a 32-bit integer.

        Args:
            ip (str | float | int): IP address to convert.

        Returns:
            int: Converted integer representation, -1 if invalid or missing.
        """
        if isinstance(ip, (int, float)):
            if pd.isna(ip):
                return -1
            return int(ip)

        if isinstance(ip, str):
            try:
                parts = ip.strip().split(".")
                if len(parts) != 4:
                    return -1
                value = 0
                for part in parts:
                    value = value * 256 + int(part)
                return value
            except (ValueError, AttributeError):
                return -1

        return -1

    def ip_series_to_int(self, ip_series: pd.Series) -> pd.Series:
        """
        Convert a pandas Series of IPs to integer representation.

        Args:
            ip_series (pd.Series): Series of IP addresses.

        Returns:
            pd.Series: Integer representation of IPs.
        """
        if pd.api.types.is_numeric_dtype(ip_series):
            return ip_series.fillna(-1).astype(np.int64)
        return ip_series.apply(self.ip_to_int)

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
        df["lower_bound_ip_address"] = pd.to_numeric(
            df["lower_bound_ip_address"], errors="coerce"
        ).fillna(0).astype(np.int64)
        df["upper_bound_ip_address"] = pd.to_numeric(
            df["upper_bound_ip_address"], errors="coerce"
        ).fillna(0).astype(np.int64)
        df = df.sort_values("lower_bound_ip_address").reset_index(drop=True)
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
            ip_ranges[["lower_bound_ip_address", "upper_bound_ip_address", "country"]],
            left_on="ip_int",
            right_on="lower_bound_ip_address",
            direction="backward",
        )

        # Validate upper bound + invalid IPs
        invalid_mask = (
            (merged["ip_int"] > merged["upper_bound_ip_address"])
            | (merged["lower_bound_ip_address"].isna())
            | (merged["ip_int"] < 0)
        )
        merged.loc[invalid_mask, "country"] = "Unknown"

        # Cleanup temporary columns
        merged = (
            merged.sort_values("index")
            .drop(
                columns=[
                    "index",
                    "ip_int",
                    "lower_bound_ip_address",
                    "upper_bound_ip_address",
                ]
            )
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
        stats = self.transactions_df.groupby("country").agg(
            total_transactions=(self.target_col, "count"),
            fraud_count=(self.target_col, "sum"),
        ).reset_index()

        stats["fraud_rate"] = stats["fraud_count"] / stats["total_transactions"]
        stats = stats.sort_values("fraud_count", ascending=False).reset_index(drop=True)
        return stats
