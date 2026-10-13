import pandas as pd
import numpy as np

# Base tickers selected by correlation + permutation-importance analysis
BASE_FEATURES = ['DXY', 'JPY', 'VIX', 'GTITL2YR', 'XAU BGNL']

# Final engineered feature set fed into both pipeline stages
FEATURE_COLS = [
    'DXY', 'JPY', 'VIX', 'GTITL2YR', 'XAU BGNL',
    'XAU BGNL_cross', 'VIX_cross', 'USD_Over_100',
]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['Data'])
    df = df.sort_values('Data').reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 5w/20w rolling averages, MA-cross signals for XAU BGNL and VIX,
    and a USD_Over_100 flag. Drops any row where the rolling windows are not
    yet fully populated (first 19 rows of any window). Returns only FEATURE_COLS
    (plus Data and Y if present).
    """
    df = df.copy()
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values('Data').reset_index(drop=True)

    rolling_cols = []
    for col in BASE_FEATURES:
        r5, r20 = f'{col}_rolling_avg_5', f'{col}_rolling_avg_20'
        df[r5] = df[col].rolling(window=5).mean()
        df[r20] = df[col].rolling(window=20).mean()
        rolling_cols.extend([r5, r20])

    # Drop rows before rolling windows are fully populated; must happen before
    # computing crosses so NaN comparisons don't silently produce False (0).
    df = df.dropna(subset=rolling_cols).reset_index(drop=True)

    for col in ['XAU BGNL', 'VIX']:
        df[f'{col}_cross'] = (
            df[f'{col}_rolling_avg_5'] > df[f'{col}_rolling_avg_20']
        ).astype(int)

    df['USD_Over_100'] = (df['DXY'] > 100).astype(int)

    keep = ['Data'] + (['Y'] if 'Y' in df.columns else []) + FEATURE_COLS
    return df[keep].reset_index(drop=True)
