import pandas as pd
import numpy as np

def validate_timeseries(ts_in):
    # Check it's a pandas Series or DataFrame
    if not isinstance(ts_in, (pd.Series, pd.DataFrame)):
        raise TypeError("Input must be a pandas Series or DataFrame.")

    # If DataFrame, check it has exactly one column
    if isinstance(ts_in, pd.DataFrame) and ts_in.shape[1] != 1:
        raise TypeError("DataFrame must have exactly one column.")

    # Check index type is DatetimeIndex
    if not isinstance(ts_in.index, pd.DatetimeIndex):
        raise TypeError("Index must be a pandas DatetimeIndex.")

    # Check for missing values
    if ts_in.isnull().any().any():
        raise ValueError("Time series contains missing values.")

    # Check for regular time intervals
    time_diffs = ts_in.index.to_series().diff().dropna()
    if not np.allclose(time_diffs, time_diffs.iloc[0]):
        raise ValueError("Time series does not have regular time intervals.")

    # Check for monotonic index
    if not ts_in.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted in ascending order.")

    # Check for non-empty input
    if ts_in.empty:
        raise ValueError("Time series is empty.")
