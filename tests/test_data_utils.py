import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.cleaner import TransactionDataCleaner
from src.data.loader import FraudDataLoader, CreditCardDataLoader, IPCountryLoader
from src.data.merger import GeoDataMerger

# -----------------------------
# TransactionDataCleaner tests
# -----------------------------
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "user_id": [1, 2, 3, 4, 1],
        "age": [25, 30, 35, 40, 25],
        "purchase_value": [100, 200, 150, 250, 100],
        "device_id": ["d1", "d2", "d3", "d4", "d1"],
        "source": ["SEO", "Ads", "SEO", "Ads", "SEO"],
        "browser": ["Chrome", "Safari", "Firefox", "Edge", "Chrome"],
        "sex": ["M", "F", "M", "F", "M"],
        "ip_address": ["192.168.0.1", "10.0.0.1", "8.8.8.8", "1.1.1.1", "192.168.0.1"],
        "class": [0, 1, 0, 0, 0]
    })

def test_remove_duplicates(sample_df):
    cleaner = TransactionDataCleaner(sample_df)
    df_cleaned = cleaner.remove_duplicates()
    assert df_cleaned.shape[0] == 4  # one duplicate removed

def test_handle_missing_values_drop(sample_df):
    df_with_na = sample_df.copy()
    df_with_na.loc[0, "purchase_value"] = None
    cleaner = TransactionDataCleaner(df_with_na)
    df_cleaned = cleaner.handle_missing_values(strategy="drop")
    assert df_cleaned.isna().sum().sum() == 0

def test_handle_missing_values_fill_zero(sample_df):
    df_with_na = sample_df.copy()
    df_with_na.loc[0, "purchase_value"] = None
    cleaner = TransactionDataCleaner(df_with_na)
    df_filled = cleaner.handle_missing_values(strategy="fill_zero")
    assert df_filled.isna().sum().sum() == 0

def test_correct_dtypes(sample_df):
    cleaner = TransactionDataCleaner(sample_df)
    df_corrected = cleaner.correct_dtypes()
    assert df_corrected["age"].dtype == int

# -----------------------------
# FraudDataLoader / CreditCardDataLoader / IPCountryLoader tests
# -----------------------------
@pytest.fixture
def fraud_csv_data():
    return """user_id,signup_time,purchase_time,purchase_value,device_id,source,browser,sex,age,ip_address,class
1,2025-01-01 10:00:00,2025-01-01 12:00:00,100,d1,SEO,Chrome,M,25,192.168.0.1,0
2,2025-02-01 09:00:00,2025-02-01 10:00:00,200,d2,Ads,Safari,F,30,10.0.0.1,1"""

def test_fraud_data_loader(tmp_path, fraud_csv_data):
    file_path = tmp_path / "fraud.csv"
    file_path.write_text(fraud_csv_data)
    loader = FraudDataLoader(str(file_path))
    df = loader.load()
    assert "signup_time" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["signup_time"])

@pytest.fixture
def credit_csv_data():
    # Minimal V1-V3 for testing; normally there are V1-V28
    return """Time,V1,V2,V3,Amount,Class
0,0.1,0.2,0.3,100,0
60,0.4,0.5,0.6,200,1"""

def test_credit_card_loader(tmp_path, credit_csv_data):
    file_path = tmp_path / "credit.csv"
    file_path.write_text(credit_csv_data)
    loader = CreditCardDataLoader(str(file_path))
    df = loader.load()
    assert "Amount" in df.columns
    assert df.shape[0] == 2

def test_ip_country_loader(tmp_path):
    ip_csv = "lower_bound_ip_address,upper_bound_ip_address,country\n0,16777215,USA\n16777216,33554431,CAN"
    file_path = tmp_path / "ip.csv"
    file_path.write_text(ip_csv)
    loader = IPCountryLoader(str(file_path))
    df = loader.load()
    assert "country" in df.columns
    assert df.shape[0] == 2

# -----------------------------
# GeoDataMerger tests
# -----------------------------
@pytest.fixture
def transactions_df():
    return pd.DataFrame({
        "ip_address": ["0.0.0.1", "1.0.0.1", "255.255.255.255"],
        "class": [0, 1, 0]
    })

@pytest.fixture
def ip_df():
    return pd.DataFrame({
        "lower_bound_ip_address": [0, 16777216],
        "upper_bound_ip_address": [16777215, 33554431],
        "country": ["USA", "CAN"]
    })

def test_ip_to_int_and_series(transactions_df):
    merger = GeoDataMerger(transactions_df, pd.DataFrame())
    assert merger.ip_to_int("1.2.3.4") == 16909060
    series_int = merger.ip_series_to_int(transactions_df["ip_address"])
    assert all(isinstance(x, (int, np.integer)) for x in series_int)

def test_merge_country(transactions_df, ip_df):
    merger = GeoDataMerger(transactions_df, ip_df)
    merged_df = merger.merge_country()
    assert "country" in merged_df.columns
    assert merged_df["country"].iloc[0] in ["USA", "CAN", "Unknown"]

def test_get_summary(transactions_df, ip_df):
    merger = GeoDataMerger(transactions_df, ip_df)
    merger.merge_country()
    summary = merger.get_summary()
    assert "fraud_rate" in summary.columns
    assert summary["fraud_rate"].max() <= 1.0
