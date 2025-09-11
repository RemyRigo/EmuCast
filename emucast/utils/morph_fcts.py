import numpy as np

def morph_nrmse(y_true, y_pred, target):

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mask = np.abs(y_true) >= 1e-3

    alfa = target / 100.0 * np.mean(y_true[mask]) / np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)) - 1.0

    return y_pred + alfa * (y_pred - y_true)


def morph_nmae(y_true, y_pred, target):

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mask = np.abs(y_true) >= 1e-3

    alfa = target / 100.0 * np.mean(y_true[mask]) / np.mean(np.abs(y_true[mask] - y_pred[mask])) - 1.0

    return y_pred + alfa * (y_pred - y_true)


def morph_eof(y_true, y_pred, target):

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    alfa = target/ 100.0 * np.sqrt(np.sum((y_true - np.mean(y_true)) ** 2)) / np.sqrt(np.sum((y_pred -y_true) ** 2))-1.0

    return y_pred + alfa * (y_pred - y_true)