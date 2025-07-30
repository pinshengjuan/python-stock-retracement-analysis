import pandas as pd

def n_day_average_volume(data: pd.DataFrame, periods: int) -> pd.Series:
    """
    Calculate the n-day average volume for the stock.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        periods (int): Number of days for the average volume.
    Returns:
        pd.Series: Series containing the n-day average volume.
    """
    volume: pd.Series = data['Volume']
    return volume.rolling(window=periods).mean()