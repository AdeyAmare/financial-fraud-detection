from typing import Union
import pandas as pd
import numpy as np


class GeoDataMerger:
    """
    Class-based wrapper for IP-to-country geolocation utilities.
    Behavior is identical to the functional implementation.
    """

    def __init__(
        self,
        transactions_df: pd.DataFrame,
        ip_df: pd.DataFrame,
        ip_column: str = "ip_address",
        target_col: str = "class",
    ):
        self.transactions_df = transactions_df.copy()
        self.ip_df = ip_df.copy()
        self.ip_column = ip_column
        self.target_col = target_col

    # -----------------------------
    # IP conversion utilities
    # -----------------------------
    @staticmethod
    def ip_to_int(ip: Union[str, float, int]) -> int:
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
        if pd.api.types.is_numeric_dtype(ip_series):
            return ip_series.fillna(-1).astype(np.int64)

        return ip_series.apply(self.ip_to_int)

    # -----------------------------
    # IP range preparation
    # -----------------------------
    def prepare_ip_ranges(self) -> pd.DataFrame:
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
        df = self.transactions_df.copy()

        # Convert IPs
        df["ip_int"] = self.ip_series_to_int(df[self.ip_column])

        # Prepare ranges
        ip_ranges = self.prepare_ip_ranges()

        # Sort for merge_asof
        df_sorted = df.sort_values("ip_int").reset_index()

        merged = pd.merge_asof(
            df_sorted,
            ip_ranges[
                ["lower_bound_ip_address", "upper_bound_ip_address", "country"]
            ],
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

        # Cleanup
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
        stats = self.transactions_df.groupby("country").agg(
            total_transactions=(self.target_col, "count"),
            fraud_count=(self.target_col, "sum"),
        ).reset_index()

        stats["fraud_rate"] = (
            stats["fraud_count"] / stats["total_transactions"]
        )

        stats = stats.sort_values("fraud_count", ascending=False).reset_index(drop=True)
        return stats
