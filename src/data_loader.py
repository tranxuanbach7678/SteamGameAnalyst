"""
Data loader module for the Steam Games Dataset.
Handles robust path resolution, raw CSV parsing anomaly detection and correction.
Course: CSC17104 - Programming for Data Science, HCMUS.
"""

import os
from pathlib import Path
import pandas as pd

# The raw Steam Games CSV has 40 values per row, but the original header merged
# 'Discount' and 'DLC count' into 'DiscountDLC count' (resulting in only 39 headers).
# Passing this standardized 40-column list restores perfect column alignment.
CORRECTED_COLUMNS = [
    'AppID', 'Name', 'Release date', 'Estimated owners', 'Peak CCU',
    'Required age', 'Price', 'Discount', 'DLC count', 'About the game',
    'Supported languages', 'Full audio languages', 'Reviews', 'Header image',
    'Website', 'Support url', 'Support email', 'Windows', 'Mac', 'Linux',
    'Metacritic score', 'Metacritic url', 'User score', 'Positive',
    'Negative', 'Score rank', 'Achievements', 'Recommendations', 'Notes',
    'Average playtime forever', 'Average playtime two weeks',
    'Median playtime forever', 'Median playtime two weeks', 'Developers',
    'Publishers', 'Categories', 'Genres', 'Tags', 'Screenshots', 'Movies'
]


def find_data_file(filename: str = 'games.csv') -> str:
    """
    Search across potential relative paths to locate the dataset file.
    Works whether running from the repository root, src/, or notebooks/.
    """
    candidates = [
        Path(filename),
        Path('data') / filename,
        Path('data') / 'raw' / filename,
        Path('..') / 'data' / filename,
        Path('..') / 'data' / 'raw' / filename,
        Path('SteamGameAnalyst') / 'data' / filename,
        Path('SteamGameAnalyst') / 'data' / 'raw' / filename,
        Path('..') / 'SteamGameAnalyst' / 'data' / filename,
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return str(p.resolve())
    raise FileNotFoundError(
        f"Could not locate '{filename}' in standard data folders. "
        "Please ensure the dataset is placed under data/ or data/raw/."
    )


def load_steam_raw_data(data_path: str = None, nrows: int = None, low_memory: bool = False) -> pd.DataFrame:
    """
    Load raw Steam Games CSV into a pandas DataFrame with corrected header mapping.

    Parameters:
    -----------
    data_path : str, optional
        Explicit path to the CSV file. If None, automatically searches candidate folders.
    nrows : int, optional
        Number of rows to read (useful for rapid exploration / testing).
    low_memory : bool, default False
        Passed to pd.read_csv.

    Returns:
    --------
    pd.DataFrame : Cleanly loaded DataFrame with 40 aligned columns.
    """
    if data_path is None:
        data_path = find_data_file('games.csv')

    print(f"Loading Steam Games dataset from: {data_path}")
    df = pd.read_csv(
        data_path,
        names=CORRECTED_COLUMNS,
        header=0,
        nrows=nrows,
        low_memory=low_memory,
        encoding='utf-8',
        on_bad_lines='skip'
    )
    print(f"Successfully loaded {len(df):,} rows and {df.shape[1]} columns.")
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Memory footprint: {mem_mb:.2f} MB")
    return df

