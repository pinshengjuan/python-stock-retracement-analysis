import pandas as pd

def check_sma_cross(data: pd.DataFrame, sma_df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for SMA cross signals in the data.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        sma_df (pd.DataFrame): DataFrame containing SMA values.
    Returns:
        pd.DataFrame: DataFrame containing signals for SMA crosses.
    """
    close = data['Close']
    signals = pd.DataFrame(index=data.index, columns=sma_df.columns)
    for period in sma_df.columns:
        sma = sma_df[period]
        prev_close = close.shift(1)
        prev_sma = sma.shift(1)
        signals[period] = 'below SMA'
        golden_cross = (close > sma) & (prev_close <= prev_sma)
        death_cross = (close < sma) & (prev_close >= prev_sma)
        above_sma = (close > sma) & (prev_close > prev_sma)
        signals.loc[golden_cross, period] = 'golden cross'
        signals.loc[death_cross, period] = 'death cross'
        signals.loc[above_sma, period] = 'above SMA'
    return signals

def check_two_sma_relation(sma_df: pd.DataFrame, short_period: str, long_period: str) -> pd.DataFrame:
    """
    Check the relationship between two SMAs.
    Args:
        sma_df (pd.DataFrame): DataFrame containing SMA values.
        short_period (str): Shorter SMA period.
        long_period (str): Longer SMA period.
    Returns:
        pd.DataFrame: DataFrame containing signals for the relationship between two SMAs.
    """
    signals = pd.DataFrame(index=sma_df.index)
    short_sma_close = sma_df[short_period]
    short_sma_prev_close = short_sma_close.shift(1)
    long_sma_close = sma_df[long_period]
    long_sma_prev_close = long_sma_close.shift(1)
    signals['signal'] = 'below long-term SMA'
    golden_cross = (short_sma_close > long_sma_close) & (short_sma_prev_close <= long_sma_prev_close)
    death_cross = (short_sma_close < long_sma_close) & (short_sma_prev_close >= long_sma_prev_close)
    above_sma = (short_sma_close > long_sma_close) & (short_sma_prev_close > long_sma_prev_close)
    signals.loc[golden_cross, 'signal'] = 'golden cross'
    signals.loc[death_cross, 'signal'] = 'death cross'
    signals.loc[above_sma, 'signal'] = 'above long-term SMA'
    return signals

def is_below_sma(data: pd.DataFrame, sma_df: pd.DataFrame) -> pd.DataFrame:
    """
    Check if the stock price is below the SMAs.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        sma_df (pd.DataFrame): DataFrame containing SMA values.
    Returns:
        pd.DataFrame: DataFrame containing boolean values indicating if the stock price is below the SMAs.
    """
    close = data['Close']
    signals = pd.DataFrame(index=data.index, columns=sma_df.columns)
    for period in sma_df.columns:
        sma = sma_df[period]
        signals[period] = close < sma
    return signals