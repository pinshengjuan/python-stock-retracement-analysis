import pandas as pd

def fib_retracement(data: pd.DataFrame, periods: int = 90) -> pd.Series:
    """
    Calculate Fibonacci retracement levels for the stock.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        periods (int): Number of periods for the Fibonacci retracement calculation (default is 90).
    Returns:
        pd.Series: Series containing the Fibonacci retracement levels.
    """
    if len(data) <= periods:
        raise ValueError(f"Data length ({len(data)}) must be > periods ({periods})")
    close: pd.Series = data['Close']
    max_price: pd.Series = close.rolling(window=periods).max()
    min_price: pd.Series = close.rolling(window=periods).min()
    diff: pd.Series = max_price - min_price
    levels = {
        '100.0%': max_price,
        '61.8%': max_price - (diff * 0.382),
        '50.0%': max_price - (diff * 0.5),
        '38.2%': max_price - (diff * 0.618),
        '0.0%': min_price
    }
    level_values = list(levels.values())
    level_keys = list(levels.keys())
    position: pd.Series = pd.Series(index=close.index, dtype=str)
    for i in range(len(close)):
        price = close.iloc[i]
        if pd.isna(max_price.iloc[i]) or pd.isna(min_price.iloc[i]):
            continue
        if price > max_price.iloc[i]:
            position.iloc[i] = 'Above 100.0%'
            continue
        elif price == max_price.iloc[i]:
            position.iloc[i] = 'Equals 100.0%'
            continue
        elif price < min_price.iloc[i]:
            position.iloc[i] = 'Below 0.0%'
            continue
        elif price == min_price.iloc[i]:
            position.iloc[i] = 'Equals 0.0%'
            continue
        for j in range(len(levels) - 1):
            level_high = level_values[j].iloc[i]
            level_second_high = level_values[j + 1].iloc[i]
            if level_second_high < price < level_high:
                position.iloc[i] = f'Between {level_keys[j+1]} and {level_keys[j]}'
    return position
