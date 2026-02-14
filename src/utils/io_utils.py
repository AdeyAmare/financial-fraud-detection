import pandas as pd
from pathlib import Path
import logging
from typing import Literal

# Configure module-level logging
logger = logging.getLogger("DataIO")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def save_dataframe(
    df: pd.DataFrame,
    path: str,
    index: bool = False,
    file_format: Literal["csv", "parquet"] = "csv",
    overwrite: bool = True,
) -> None:
    """
    Save a pandas DataFrame to disk safely.

    Features:
        - Creates parent directories if they do not exist.
        - Supports CSV (default) or Parquet formats.
        - Optional index column.
        - Optional overwrite protection.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        path (str): Destination file path.
        index (bool, optional): Whether to write row names (index). Defaults to False.
        file_format (str, optional): 'csv' or 'parquet'. Defaults to 'csv'.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to True.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If path is invalid or DataFrame is empty.
        FileExistsError: If file exists and overwrite is False.
        OSError: If saving the file fails due to OS-level issues.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)}")
    if df.empty:
        raise ValueError("Cannot save an empty DataFrame")
    if not path or not isinstance(path, str):
        raise ValueError(f"Invalid path: {path}")

    file_path = Path(path)

    if file_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists and overwrite is False: {file_path}")

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_format == "csv":
            df.to_csv(file_path, index=index)
        elif file_format == "parquet":
            df.to_parquet(file_path, index=index)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        logger.info(f"[IO] Data saved successfully to {file_path} ({file_format})")

    except OSError as e:
        logger.error(f"Failed to save DataFrame to {file_path}: {e}")
        raise
