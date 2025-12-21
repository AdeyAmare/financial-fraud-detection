import pandas as pd
from pathlib import Path


def save_dataframe(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """
    Save a pandas DataFrame to a CSV file safely.

    Features:
    - Creates parent directories if they do not exist.
    - Supports optional inclusion of the index column.

    Args:
        df (pd.DataFrame): DataFrame to save.
        path (str): Destination file path for the CSV.
        index (bool, optional): Whether to write row names (index). Defaults to False.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If the path is empty or invalid.
        OSError: If saving the file fails due to OS-level issues.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)}")
    if not path or not isinstance(path, str):
        raise ValueError(f"Invalid path: {path}")

    file_path = Path(path)

    try:
        # Create parent directories if they do not exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Save the DataFrame
        df.to_csv(file_path, index=index)
        print(f"[IO] Data saved to {file_path}")
    except OSError as e:
        raise OSError(f"Failed to save DataFrame to {file_path}: {e}")
