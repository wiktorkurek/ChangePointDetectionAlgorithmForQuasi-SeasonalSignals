import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
import os

def split_signal(arr, percentile = 95, season = None, autocorrelation_lags = 100):
    acorr = sm.tsa.acf(arr, nlags = autocorrelation_lags)
    
    if season is None:
        maks = 0
        season = 0
        for i in range(1, acorr.shape[0] - 1):
            if (acorr[i] > acorr[i-1]) and (acorr[i] > acorr[i+1]) and (acorr[i] > maks):
                maks = acorr[i]
                season = i
    
    matrix = arr.reshape(-1, season)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    normalized = (matrix - means) / stds
    rmse = ((normalized[:-1] - normalized[1:]) ** 2).sum(axis=1) ** 0.5
    p = np.percentile(rmse, percentile)
    indices = np.arange(rmse.shape[0])[rmse > p]
    
    return indices, season

def plot_splits(arr, indices, season, folder_path):
    matrix = arr.reshape(-1, season)
    for i in indices:
        plt.plot(matrix[i], label=f'season {i}')
        plt.plot(matrix[i+1], label=f"season {i+1}")
        plt.legend()
        plt.savefig(os.path.join(folder_path, f"seasons_{i}_{i+1}.png"))
        plt.clf()