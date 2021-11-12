import pandas as pd
import numpy as np

def realized_volatility(series):
    return np.sqrt(np.sum(series**2))

def log_return(series: np.ndarray):
    return np.log(series).diff()

def log_return_df2(series: np.ndarray):
    return np.log(series).diff(2)
