import pandas as pd

def get_current_price(data_set: pd.DataFrame) -> float:
    """
    Get the current price of the stock from the DataFrame.
    Args:
        data_set (pd.DataFrame): DataFrame containing stock data.
    Returns:
        float: Current stock price.
    """
    return data_set['Close'].iloc[-1]

def get_change_percentage(data_set: pd.DataFrame) -> str:
    """
    Calculate the percentage change of the stock price. 
    Args:
        data_set (pd.DataFrame): DataFrame containing stock data.
    Returns:
        str: Percentage change formatted as a string.
    """
    change_percentage = data_set['Close'].pct_change().iloc[-1] * 100
    rounfded_change = round(change_percentage, 2)
    return f"{rounfded_change}%"

def get_latest_volume(data_set: pd.DataFrame) -> int:
    """
    Get the latest volume of the stock from the DataFrame.
    Args:
        data_set (pd.DataFrame): DataFrame containing stock data.
    Returns:
        int: Latest stock volume.
    """
    return data_set['Volume'].iloc[-1]
