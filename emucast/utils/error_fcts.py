import numpy as np


# =========================================================
# Utility internal function
# =========================================================
def _prepare_inputs(y_true, y_pred):
    """
    Ensure shapes:
    - y_true: (T, 1)
    - y_pred: (T, N)
    """
    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    return y_true, y_pred


# =========================================================
# NMAE
# =========================================================
def nmae(y_true, y_pred):
    """
    Normalized Mean Absolute Error (%)

    Supports:
    - y_true: (T,)
    - y_pred: (T,) or (T, N)

    Returns:
    - scalar if N=1
    - array (N,) if multiple scenarios
    """
    y_true, y_pred = _prepare_inputs(y_true, y_pred)

    mask = np.abs(y_true[:, 0]) >= 1e-3

    if not np.any(mask):
        return np.nan if y_pred.shape[1] == 1 else np.full(y_pred.shape[1], np.nan)

    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask, :]

    numerator = np.mean(np.abs(y_true_masked - y_pred_masked), axis=0)
    denom = max(1e-6, abs(np.mean(y_true_masked)))

    result = numerator / denom * 100.0

    return result[0] if result.shape[0] == 1 else result


# =========================================================
# NRMSE
# =========================================================
def nrmse(y_true, y_pred):
    """
    Normalized Root Mean Square Error (%)
    """
    y_true, y_pred = _prepare_inputs(y_true, y_pred)

    mask = np.abs(y_true[:, 0]) >= 1e-3

    if not np.any(mask):
        return np.nan if y_pred.shape[1] == 1 else np.full(y_pred.shape[1], np.nan)

    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask, :]

    mse = np.mean((y_true_masked - y_pred_masked) ** 2, axis=0)
    denom = max(1e-6, abs(np.mean(y_true_masked)))

    result = np.sqrt(mse) / denom * 100.0

    return result[0] if result.shape[0] == 1 else result


# =========================================================
# EOF
# =========================================================
def eof(y_true, y_pred):
    """
    Error of Fit (%)
    """
    y_true, y_pred = _prepare_inputs(y_true, y_pred)

    num = np.sqrt(np.sum((y_pred - y_true) ** 2, axis=0))

    denom = np.sqrt(np.sum((y_true - np.mean(y_true)) ** 2))
    denom = max(1e-6, denom)

    result = num / denom * 100.0

    return result[0] if result.shape[0] == 1 else result