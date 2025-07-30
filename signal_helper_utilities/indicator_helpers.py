import pandas as pd
import numpy as np

def price_sma_range_percentage(data: pd.DataFrame, sma_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the percentage difference between the stock price and SMAs.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        sma_df (pd.DataFrame): DataFrame containing SMA values.
    Returns:
        pd.DataFrame: DataFrame containing percentage differences between stock price and SMAs.
    """
    close = data['Close']
    signals = pd.DataFrame(index=data.index, columns=sma_df.columns)
    for period in sma_df.columns:
        sma = sma_df[period]
        percentage = ((sma - close) / sma) * 100
        rounded = percentage.round(2).abs().astype(str)
        signals[period] = np.where(
            percentage == 0,
            "equals",
            np.where(
                percentage > 0,
                "Price " + rounded + "% below sma(" + period + ")",
                "Price " + rounded + "% above sma(" + period + ")",
            )
        )
    return signals