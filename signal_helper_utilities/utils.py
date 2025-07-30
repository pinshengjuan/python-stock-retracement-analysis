import os
import threading
import sys
import time
import pandas as pd
import numpy as np
from typing import Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def spinner(msg="Generating analysis results"):
    """
    Display a spinner while processing.
    This function runs in a separate thread to avoid blocking the main thread.
    """
    spinner_chars = ['|', '/', '-', '\\']
    idx = 0
    while not getattr(threading.current_thread(), "stop", False):
        sys.stdout.write(f'\r{msg} {spinner_chars[idx % len(spinner_chars)]}')
        sys.stdout.flush()
        time.sleep(0.1)
        idx += 1
    sys.stdout.write(f'\r{msg} Done!\n')
    sys.stdout.flush()

def load_config() -> dict[str, Any]:
    """
    Load configuration settings from environment variables or defaults.
    Returns:
        dict: Configuration settings.
    """
    return {
        'TICKER_LIST': os.getenv('TICKER_LIST', 'GOOGL'),
        'DAYS': int(os.getenv('DAYS', 365)),
        'VOLUME_AVERAGE_PERIOD': int(os.getenv('VOLUME_AVERAGE_PERIOD', 14)),
        'RSI_PERIOD_SHORT': int(os.getenv('RSI_PERIOD_SHORT', 14)),
        'RSI_PERIOD_LONG': int(os.getenv('RSI_PERIOD_LONG', 50)),
        'ADX_PERIOD_SHORT': int(os.getenv('ADX_PERIOD_SHORT', 14)),
        'ADX_PERIOD_LONG': int(os.getenv('ADX_PERIOD_LONG', 50)),
        'AROON_PERIOD': int(os.getenv('AROON_PERIOD', 14)),  # short-term=14; long-term=25
        'BOLLINGER_PERIOD': int(os.getenv('BOLLINGER_PERIOD', 20)),  # short-term=20; long-term=30
        'FIBONACCI_PERIOD': int(os.getenv('FIBONACCI_PERIOD', 90)),
    }

def wilder_smoothing(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Apply Wilder's smoothing to a series.
    Args:
        series (pd.Series): Series to smooth.
        period (int): Period for smoothing.
    Returns:
        pd.Series: Smoothed series.
    """
    smoothed = np.zeros(len(series))
    smoothed[period-1] = series[:period].mean()
    for i in range(period, len(series)):
        smoothed[i] = (smoothed[i-1] * (period-1) + series.iloc[i]) / period
    smoothed[:period-1] = np.nan
    return pd.Series(smoothed, index=series.index)

def write_to_file(data_set: Any, filename: str, mode: str = 'w') -> None:
    """
    Write data to a file.
    Args:
        data_set (Any): Data to write to the file.
        filename (str): Name of the file to write to.
        mode (str): Mode in which to open the file (default is 'w').
    """
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/data/'
    if not os.path.isdir(parent_dir):
        os.mkdir(parent_dir)
    filename: str = parent_dir + filename
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    with open(filename, mode, encoding='utf8') as f:
        f.write(str(data_set))