import numpy as np

def nmae(y_true, y_pred):

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mask = np.abs(y_true) >= 1e-3
    if not np.any(mask):
        return np.nan

    return np.mean(np.abs(y_true[mask] - y_pred[mask])) / max(1e-6,abs(np.mean(y_true[mask]))) * 100.0


def nrmse(y_true, y_pred):

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mask = np.abs(y_true) >= 1e-3
    if not np.any(mask):
        return np.nan

    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)) / max(1e-6,abs(np.mean(y_true[mask]))) * 100.0

def eof(y_true, y_pred):

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    return np.sqrt(np.sum((y_pred - y_true) ** 2))/max(1e-6, np.sqrt(np.sum((y_true - np.mean(
        y_true))**2)))*100.0


