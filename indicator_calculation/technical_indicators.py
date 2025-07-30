import pandas as pd
import numpy as np
from typing import Tuple
from signal_helper_utilities.utils import wilder_smoothing

def sma(data: pd.DataFrame, periods: int, days_back: int = 1) -> float:
    """
    Calculate the Simple Moving Average (SMA) for a given number of periods.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        periods (int): Number of periods for the SMA.
        days_back (int): Number of days back to calculate the SMA (default is 1).
    Returns:
        float: The calculated SMA value rounded to 2 decimal places.
    """
    close: pd.Series = data['Close']
    return np.round(close.rolling(window=periods).mean().iloc[-days_back], 2)

def sma_series(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Simple Moving Averages (SMA) for multiple periods.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
    Returns:
        pd.DataFrame: DataFrame containing SMA for various periods.
    """
    close = data['Close']
    sma_df = pd.DataFrame(index=close.index)
    periods = [
        3, 5, 7, 10, 12, 13, 15, 18, 20, 21, 26, 30, 34, 50, 60, 89, 100, 120, 144, 150, 200, 233, 240, 250
    ]
    for period in periods:
        sma_df[f'{period}'] = close.rolling(window=period).mean()
    return sma_df

def rsi(data: pd.DataFrame, periods: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for the stock.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        periods (int): Number of periods for the RSI calculation (default is 14).
    Returns:
        pd.Series: Series containing the RSI values.
    """
    if len(data) <= periods:
        raise ValueError(f"Data length ({len(data)}) must be > periods ({periods})")
    close: pd.Series = data['Close']
    delta: pd.Series = close.diff()
    gain: pd.Series = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss: pd.Series = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs: pd.Series = gain / loss
    rsi: pd.Series = 100 - (100 / (1 + rs))
    rsi = rsi.where(loss != 0, 100.0)
    return rsi

def tr(data: pd.DataFrame, periods: int = 14) -> pd.DataFrame:
    """
    Calculate the True Range (TR) for the stock.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        periods (int): Number of periods for the TR calculation (default is 14).
    Returns:
        pd.DataFrame: DataFrame containing the True Range values.
    """
    high: pd.Series = data['High']
    low: pd.Series = data['Low']
    close: pd.Series = data['Close']
    tr = pd.DataFrame(index=high.index)
    tr['h_l'] = np.round((high - low), 2)
    tr['h_pc'] = abs(np.round((high - close.shift(1)), 2))
    tr['l_pc'] = abs(np.round((low - close.shift(1)), 2))
    tr['tr'] = tr[['h_l', 'h_pc', 'l_pc']].max(axis=1)
    return tr

def atr(data: pd.DataFrame, periods: int = 14) -> pd.Series:
    """
    Calculate the Average True Range (ATR) for the stock.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        periods (int): Number of periods for the ATR calculation (default is 14).
    Returns:
        pd.Series: Series containing the Average True Range values.
    """
    tr_series: pd.DataFrame = tr(data, periods)
    atr_series: pd.Series = wilder_smoothing(tr_series['tr'], periods)
    return atr_series

def adx(data: pd.DataFrame, periods: int = 14) -> pd.Series:
    """
    Calculate the Average Directional Index (ADX) for the stock.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        periods (int): Number of periods for the ADX calculation (default is 14).
    Returns:
        pd.Series: Series containing the Average Directional Index values.
    """
    high: pd.Series = data['High']
    low: pd.Series = data['Low']
    close: pd.Series = data['Close']
    dm_plus: pd.Series = high - high.shift(1)
    dm_minus: pd.Series = low.shift(1) - low
    dm_plus = dm_plus.where((dm_plus > 0) & (dm_plus > dm_minus), 0)
    dm_minus = dm_minus.where((dm_minus > 0) & (dm_minus > dm_plus), 0)
    smoothed_dm_plus: pd.Series = wilder_smoothing(dm_plus, periods)
    smoothed_dm_minus: pd.Series = wilder_smoothing(dm_minus, periods)
    atr_series: pd.Series = atr(data, periods)
    di_plus: pd.Series = (smoothed_dm_plus / atr_series) * 100
    di_minus: pd.Series = (smoothed_dm_minus / atr_series) * 100
    dx: pd.Series = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100
    adx_series: pd.Series = wilder_smoothing(dx.fillna(0), periods)
    return adx_series

def macd(
    data: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    zero_threshold: float = 0.1,
    level_thresholds: tuple = (0.5, 0.2)
) -> pd.DataFrame:
    """
    Calculate the Moving Average Convergence Divergence (MACD) for the stock.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        fast (int): Fast EMA period (default is 12).
        slow (int): Slow EMA period (default is 26).
        signal (int): Signal line period (default is 9).
        zero_threshold (float): Threshold for near-zero MACD values (default is 0.1).
        level_thresholds (tuple): Thresholds for classifying MACD levels (default is (0.5, 0.2)).
    Returns:
        pd.DataFrame: DataFrame containing MACD values and signals.
    """
    close: pd.Series = data['Close']
    ema_fast: pd.Series = close.ewm(span=fast, adjust=False).mean()
    ema_slow: pd.Series = close.ewm(span=slow, adjust=False).mean()
    macd: pd.Series = ema_fast - ema_slow
    signal_line: pd.Series = macd.ewm(span=signal, adjust=False).mean()
    is_macd_bullish_cross: pd.Series = (macd > signal_line) & (macd.shift() <= signal_line.shift())
    is_macd_bearish_cross: pd.Series = (macd < signal_line) & (macd.shift() >= signal_line.shift())
    is_near_zero: pd.Series = abs(macd) < zero_threshold
    strong_threshold, mild_threshold = level_thresholds
    def classify_macd_level(macd_value: float) -> str:
        if macd_value > strong_threshold:
            return "Strong Bullish"
        elif macd_value > mild_threshold:
            return "Bullish"
        elif macd_value < -strong_threshold:
            return "Strong Bearish"
        elif macd_value < -mild_threshold:
            return "Bearish"
        else:
            return "Neutral"
    macd_level: pd.Series = macd.apply(classify_macd_level)
    result = pd.DataFrame({
        'is_macd_bullish_cross': is_macd_bullish_cross,
        'is_macd_bearish_cross': is_macd_bearish_cross,
        'is_near_zero': is_near_zero,
        'macd_level': macd_level,
        'macd_value': macd,
        'signal_line': signal_line,
    })
    return result

def aroon(data: pd.DataFrame, periods: int = 25) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate the Aroon indicator for the stock.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        periods (int): Number of periods for the Aroon calculation (default is 25).
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: A tuple containing the Aroon Up, Aroon Down, and Aroon Oscillator series.
    """
    high: pd.Series = data['High']
    low: pd.Series = data['Low']
    aroon_up = pd.Series(index=high.index, dtype=float)
    aroon_down = pd.Series(index=low.index, dtype=float)
    aroon_osc = pd.Series(index=high.index, dtype=float)
    for i in range(periods, len(high)):
        high_window = high[i-periods:i+1]
        low_window = low[i-periods:i+1]
        high_idx = high_window.idxmax()
        low_idx = low_window.idxmin()
        periods_since_high = i - high.index.get_loc(high_idx)
        periods_since_low = i - low.index.get_loc(low_idx)
        aroon_up.iloc[i] = ((periods - periods_since_high) / periods) * 100
        aroon_down.iloc[i] = ((periods - periods_since_low) / periods) * 100
        aroon_osc.iloc[i] = aroon_up.iloc[i] - aroon_down.iloc[i]
    return aroon_up, aroon_down, aroon_osc

def bollinger_bands(
    data: pd.DataFrame,
    window: int = 20,
    k: float = 2,
    return_all_zones: bool = False
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands for the stock.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        window (int): Window size for the Bollinger Bands (default is 20).
        k (float): Number of standard deviations for the bands (default is 2).
        return_all_zones (bool): Whether to return all zones or just the position (default is False).
    Returns:
        pd.DataFrame: DataFrame containing the Bollinger Bands and position.
    """
    if window <= 0 or k <= 0:
        raise ValueError("Window and k must be positive")
    close: pd.Series = data['Close']
    sma: pd.Series = close.rolling(window=window).mean()
    std_n: pd.Series = close.rolling(window=window).std()
    upper_band: pd.Series = sma + (std_n * k)
    lower_band: pd.Series = sma - (std_n * k)
    upper_0_3: pd.Series = upper_band * 0.97
    upper_3_5: pd.Series = upper_band * 0.95
    upper_5_10: pd.Series = upper_band * 0.90
    lower_0_3: pd.Series = lower_band * 1.03
    lower_3_5: pd.Series = lower_band * 1.05
    lower_5_10: pd.Series = lower_band * 1.10
    basis_plus_3: pd.Series = sma * 1.03
    basis_minus_3: pd.Series = sma * 0.97
    if return_all_zones:
        position: pd.Series = pd.Series([[] for _ in close], index=close.index, dtype=object)
    else:
        position: pd.Series = pd.Series(index=close.index, dtype=str)
    def get_zones(
        price: float,
        u: float,
        u_0_3: float,
        u_3_5: float,
        u_5_10: float,
        s: float,
        b_plus: float,
        b_minus: float,
        l_5_10: float,
        l_3_5: float,
        l_0_3: float,
        l: float
    ) -> list:
        zones = []
        if price > u:
            zones.append('Above Upper')
        elif price <= u and price >= l:
            if price >= s:
                zones.append('Between Upper and SMA')
                if price <= u and price >= u_0_3:
                    zones.append('Below Upper 0-3%')
                if price < u_0_3 and price >= u_3_5:
                    zones.append('Below Upper 3-5%')
                if price < u_3_5 and price >= u_5_10:
                    zones.append('Below Upper 5-10%')
            else:
                zones.append('Between SMA and Lower')
                if price <= l_0_3 and price > l:
                    zones.append('Above Lower 0-3%')
                if price <= l_3_5 and price > l_0_3:
                    zones.append('Above Lower 3-5%')
                if price <= l_5_10 and price > l_3_5:
                    zones.append('Above Lower 5-10%')
            if price <= b_plus and price >= b_minus:
                zones.append('Basis ±3%')
        else:
            zones.append('Below Lower')
        return zones
    for i in close.index:
        if pd.isna(close[i]) or pd.isna(sma[i]) or pd.isna(upper_band[i]) or pd.isna(lower_band[i]):
            continue
        zones = get_zones(
            close[i],
            upper_band[i], upper_0_3[i], upper_3_5[i], upper_5_10[i],
            sma[i], basis_plus_3[i], basis_minus_3[i],
            lower_5_10[i], lower_3_5[i], lower_0_3[i], lower_band[i]
        )
        if return_all_zones:
            position[i] = zones
        else:
            if 'Above Upper' in zones:
                position[i] = 'Above Upper'
            elif 'Below Lower' in zones:
                position[i] = 'Below Lower'
            elif 'Below Upper 0-3%' in zones:
                position[i] = 'Below Upper 0-3%'
            elif 'Below Upper 3-5%' in zones:
                position[i] = 'Below Upper 3-5%'
            elif 'Below Upper 5-10%' in zones:
                position[i] = 'Below Upper 5-10%'
            elif 'Above Lower 0-3%' in zones:
                position[i] = 'Above Lower 0-3%'
            elif 'Above Lower 3-5%' in zones:
                position[i] = 'Above Lower 3-5%'
            elif 'Above Lower 5-10%' in zones:
                position[i] = 'Above Lower 5-10%'
            elif 'Basis ±3%' in zones:
                position[i] = 'Basis ±3%'
            elif 'Between Upper and SMA' in zones:
                position[i] = 'Between Upper and SMA'
            else:
                position[i] = 'Between SMA and Lower'
    result = pd.DataFrame({
        'position': position,
        'upper_band': upper_band,
        'lower_band': lower_band,
    })
    return result

def stochastic(data: pd.DataFrame, k_period: int = 9, k_smooth_period: int = 3, d_period: int = 3) -> pd.DataFrame:
    """
    Calculate the Stochastic Oscillator for the stock.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        k_period (int): Period for %K calculation (default is 9).
        k_smooth_period (int): Smoothing period for %K (default is 3).
        d_period (int): Period for %D calculation (default is 3).
    Returns:
        pd.DataFrame: DataFrame containing %K and %D values.
    """
    high = data['High']
    low = data['Low']
    close = data['Close']
    result_df = pd.DataFrame(index=high.index)
    result_df['Lowest_Low'] = low.rolling(window=k_period).min()
    result_df['Highest_High'] = high.rolling(window=k_period).max()
    result_df['%K'] = 100 * (close - result_df['Lowest_Low']) / (result_df['Highest_High'] - result_df['Lowest_Low'])
    result_df['%K'] = result_df['%K'].rolling(window=k_smooth_period).mean()
    result_df['%D'] = result_df['%K'].rolling(window=d_period).mean()
    result_df = result_df.drop(['Lowest_Low', 'Highest_High'], axis=1)
    result_df = result_df.dropna()
    return result_df