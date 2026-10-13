import pandas as pd


def split_timeseries(
    df: pd.DataFrame,
    train_years: int = 9,
    val_years: int = 2,
    total_years: int = 12,
):
    """
    Chronological split for weekly data (52 rows/year).
    Data must already be sorted by date and have a 'Data' column.
    Returns (train_df, val_df, test_df) including the Data and Y columns.
    """
    df = df.sort_values('Data').reset_index(drop=True)

    train_end = train_years * 52
    val_end = (train_years + val_years) * 52
    total = total_years * 52

    if total > len(df):
        raise ValueError(
            f"Requested {total} rows ({total_years} years) but only {len(df)} rows available."
        )

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:total].copy()

    return train, val, test
