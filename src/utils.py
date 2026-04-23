import json
import pandas as pd

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types from pandas DataFrames."""

    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


def resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a different interval.
    
    interval: '4h', '1d', etc.
    """
    if df is None or df.empty:
        return df

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            df = df.set_index('time')
        elif 'Date' in df.columns:
            df = df.set_index('Date')
        df.index = pd.to_datetime(df.index)

    resampled = df.resample(interval).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled
