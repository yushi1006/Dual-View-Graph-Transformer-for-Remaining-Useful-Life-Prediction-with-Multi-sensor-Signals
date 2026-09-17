import numpy as np


def RUL_Score(y_true, y_pre):
    y_true = y_true.view(-1).cpu().detach().numpy()
    y_pre = y_pre.view(-1).cpu().detach().numpy()
    d = y_pre - y_true
    mse = np.mean(np.square(d))
    phm_score = np.sum(np.exp(-d[d < 0] / 13) - 1) + np.sum(np.exp(d[d >= 0] / 10) - 1)
    y_true = y_true[np.nonzero(y_true)]
    y_pre = y_pre[np.nonzero(y_true)]
    if len(y_true) == 0:
        mra = 0
        mape = 0
    else:
        d = y_pre - y_true
        mra = np.mean(1 - np.abs(d) / y_true)
        mape = np.mean(np.abs(d / y_true))
    return mse, phm_score, mra, mape


def AR_mse(y_true, y_pre):
    y_true = y_true.view(-1).cpu().detach().numpy()
    y_pre = y_pre.view(-1).cpu().detach().numpy()
    d = y_pre - y_true
    mse = np.mean(np.square(d))
    return mse

