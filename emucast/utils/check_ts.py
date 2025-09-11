import pandas as pd

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