# `src/utils/io_utils.py` – Data I/O Utilities

This module provides simple helper functions for **saving and managing datasets** in CSV format.

## Functions

### `save_dataframe(df, path, index=False)`
Safely saves a pandas DataFrame to a CSV file.

**Parameters:**
- `df` (`pd.DataFrame`): The DataFrame to save.
- `path` (`str`): The file path where the CSV will be saved.
- `index` (`bool`, optional): Whether to include the DataFrame index in the CSV. Defaults to `False`.

**Behavior:**
- Automatically creates parent directories if they do not exist.
- Saves the DataFrame to the specified path.
- Prints a confirmation message once the file is saved.

**Usage Example:**
```python
from src.utils.io_utils import save_dataframe
import pandas as pd

df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
save_dataframe(df, "data/processed/example.csv")
````

**Notes:**

* Designed for consistent and safe CSV export within ETL pipelines.
* Can be extended to support other file formats if needed.

