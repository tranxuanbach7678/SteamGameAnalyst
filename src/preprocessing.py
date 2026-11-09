"""
Preprocessing and feature engineering module for Steam Games Dataset.
Includes auditing functions to verify and report row counts after every transformation.
Course: CSC17104 - Programming for Data Science, HCMUS.
"""

import re
import pandas as pd
import numpy as np


def filter_with_audit(df: pd.DataFrame, condition: pd.Series, step_name: str) -> pd.DataFrame:
    """
    Filters a DataFrame while auditing and reporting row counts.
    Strictly satisfies Section 3.4 rubric requirement:
    'After every step that changes the data (filtering, merging, dropping),
    check and report the row count and explain any change.'
    """
    n_before = len(df)
    filtered_df = df[condition].copy()
    n_after = len(filtered_df)
    n_dropped = n_before - n_after
    pct_dropped = (n_dropped / n_before * 100.0) if n_before > 0 else 0.0

    print(f"=== [DATA AUDIT] {step_name} ===")
    print(f"  Rows before  : {n_before:,}")
    print(f"  Rows after   : {n_after:,}")
    print(f"  Rows removed : {n_dropped:,} ({pct_dropped:.2f}%)")
    print(f"  Rows retained: {(n_after / n_before * 100.0):.2f}%\n")
    return filtered_df


def parse_owner_midpoint(owner_series: pd.Series) -> pd.Series:
    """
    Convert SteamSpy 'min - max' owners string into estimated numerical midpoint.
    e.g., '20000 - 50000' -> 35000.
    """
    def _parse_single(val):
        if pd.isna(val) or not isinstance(val, str):
            return np.nan
        nums = re.findall(r'\d+', val.replace(',', ''))
        if len(nums) == 2:
            return (int(nums[0]) + int(nums[1])) / 2.0
        elif len(nums) == 1:
            return float(nums[0])
        return np.nan

    return owner_series.apply(_parse_single)


def clean_steam_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data types for raw columns in Steam games DataFrame.
    """
    clean_df = df.copy()

    # Convert numeric fields
    numeric_cols = [
        'Price', 'Discount', 'DLC count', 'Required age', 'Peak CCU',
        'Metacritic score', 'User score', 'Positive', 'Negative',
        'Achievements', 'Recommendations', 'Average playtime forever',
        'Median playtime forever'
    ]
    for col in numeric_cols:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').fillna(0) # chuyến giá trị về dạng số, gặp chữ thì chuyền về NAN, fill các NAN bằng 0

    # Boolean platform support
    # for col in ['Windows', 'Mac', 'Linux']:
    #     if col in clean_df.columns:
    #         clean_df[col] = clean_df[col].astype(str).str.lower().isin(['true', '1', 't'])

    # Parse Release date
    if 'Release date' in clean_df.columns:
        clean_df['Parsed_Release_Date'] = pd.to_datetime(clean_df['Release date'], errors='coerce')

    return clean_df


def extract_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive high-value domain variables for data analysis (Criteria D3).
    """
    out = df.copy()

    # Date temporal features (D2)
    if 'Parsed_Release_Date' in out.columns:
        out['Release_Year'] = out['Parsed_Release_Date'].dt.year
        out['Release_Month'] = out['Parsed_Release_Date'].dt.month
        out['Release_Quarter'] = out['Parsed_Release_Date'].dt.to_period('Q').astype(str)

    # Monetary and Business models (D3)
    out['Is_Free'] = out['Price'] == 0.0
    out['Has_DLC'] = out['DLC count'] > 0
    out['Price_Tier'] = pd.cut(
        out['Price'],
        bins=[-0.01, 0, 9.99, 29.99, float('inf')],
        labels=['Free', 'Budget (<$10)', 'Mid-tier ($10-$30)', 'Premium (>$30)']
    )

    # Review reception metrics (D3)
    out['Total_Reviews'] = out['Positive'] + out['Negative']
    out['Positive_Review_Ratio'] = np.where(
        out['Total_Reviews'] > 0,
        out['Positive'] / out['Total_Reviews'],
        np.nan
    )

    # Playtime engagement (D3)
    out['Playtime_Hours'] = out['Median playtime forever'] / 60.0

    # Cross-platform support tier (D3, D4)
    out['Platform_Count'] = out['Windows'].astype(int) + out['Mac'].astype(int) + out['Linux'].astype(int)
    out['Platform_Category'] = np.select(
        [
            out['Platform_Count'] == 1,
            out['Platform_Count'] == 2,
            out['Platform_Count'] == 3
        ],
        ['Single-Platform', 'Dual-Platform', 'All-Three Platforms'],
        default='Unknown'
    )

    # Estimated owners midpoint (D3)
    if 'Estimated owners' in out.columns:
        out['Estimated_Owners_Midpoint'] = parse_owner_midpoint(out['Estimated owners'])

    # Description length (word count) (D3)
    if 'About the game' in out.columns: # cùng logic: thiếu mô tả ≠ mô tả 0 từ
        out['Description_Word_Count'] = out['About the game'].apply(
            lambda t: np.nan if pd.isna(t) or not str(t).strip() else len(str(t).split()))

    # Supported language count (D3)
    def _count_languages(s):
        if pd.isna(s) or s.strip() in ('', '[]'):
            return np.nan                      # thiếu dữ liệu, không phải 0 hay 1
        items = re.findall(r"'([^']*)'", s)
        return len(items) if items else 1\
    
    if 'Supported languages' in out.columns:
        out['Language_Count'] = out['Supported languages'].apply(_count_languages)


    return out

