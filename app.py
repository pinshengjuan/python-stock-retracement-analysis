import threading
import numpy as np
import pandas as pd
from indicator_calculation.technical_indicators import (
    sma_series, rsi, atr, adx, macd, bollinger_bands,
    aroon, stochastic
)
from indicator_calculation.retracement_levels import (
    fib_retracement
)
from indicator_calculation.volume_utils import (
    n_day_average_volume
)
from signal_helper_utilities.indicator_signals import (
    check_sma_cross, check_two_sma_relation, is_below_sma
)
from signal_helper_utilities.indicator_helpers import (
    price_sma_range_percentage
)
from signal_helper_utilities.utils import (
    spinner, load_config, write_to_file, wilder_smoothing
)
from signal_helper_utilities.reporting_helpers import (
    get_current_price, get_change_percentage, get_latest_volume
)
from signal_helper_utilities.data_fetching import (
    get_multiple_days_price
)
from datetime import datetime

def get_single_ticker_data(ticker: str) -> pd.DataFrame:
    """
    Fetches and processes technical indicators for a single ticker.
    Args:
        ticker (str): The stock ticker symbol.
    Returns:
        pd.DataFrame: A DataFrame containing the processed technical indicators and metrics.
    """
    config = load_config()
    day_count = config["DAYS"]
    vol_avg_period = config["VOLUME_AVERAGE_PERIOD"]
    rsi_period_short = config['RSI_PERIOD_SHORT']
    rsi_period_long = config['RSI_PERIOD_LONG']
    adx_period_short = config['ADX_PERIOD_SHORT']
    adx_period_long = config['ADX_PERIOD_LONG']
    bollinger_period = config["BOLLINGER_PERIOD"]
    aroon_period = config["AROON_PERIOD"]
    fibonacci_period = config['FIBONACCI_PERIOD']

    # Load price and volume data
    data_set = get_multiple_days_price(ticker, day_count)
    current_price = np.round(get_current_price(data_set), 2)
    volume = get_latest_volume(data_set)
    daily_change_percentage = get_change_percentage(data_set)

    # Volume metrics
    n_day_avg_vol_series = n_day_average_volume(data_set, vol_avg_period)
    prev_n_day_avg_vol = n_day_avg_vol_series.iloc[-2]

    # SMA metrics
    sma_data = sma_series(data_set)
    sma_cross = check_sma_cross(data_set, sma_data)
    price_vs_sma = price_sma_range_percentage(data_set, sma_data)
    sma_relation = check_two_sma_relation(sma_data, '50', '200')

    # Technical indicators
    rsi_short = rsi(data_set, rsi_period_short)
    rsi_long = rsi(data_set, rsi_period_long)
    adx_short = adx(data_set, adx_period_short)
    adx_long = adx(data_set, adx_period_long)

    macd_short = macd(data_set)
    macd_short_bullish = 'MACD Bullish Cross' if macd_short['is_macd_bullish_cross'].iloc[-1] else ''
    macd_short_bearish = 'MACD Bearish Cross' if macd_short['is_macd_bearish_cross'].iloc[-1] else ''
    macd_short_near_zero = 'MACD Near Zero' if macd_short['is_near_zero'].iloc[-1] else ''

    macd_long = macd(data_set, 50, 20, 9)
    macd_long_bullish = 'MACD Bullish Cross' if macd_long['is_macd_bullish_cross'].iloc[-1] else ''
    macd_long_bearish = 'MACD Bearish Cross' if macd_long['is_macd_bearish_cross'].iloc[-1] else ''
    macd_long_near_zero = 'MACD Near Zero' if macd_long['is_near_zero'].iloc[-1] else ''

    bollinger = bollinger_bands(data_set, window=bollinger_period)
    fibonacci = fib_retracement(data_set, fibonacci_period)
    aroon_up, aroon_down, _ = aroon(data_set, aroon_period)
    stochastic_data = stochastic(data_set, 14)

    # Compose results
    personal_retracement = {
        "Ticker": ticker,
        "Volume": f"{volume:,}",
        "Average Volume (Prev 20 Days)": f"{int(np.round(prev_n_day_avg_vol)):,}",
        "Closed Price": current_price,
        "Change %": daily_change_percentage,
        "SMA50": np.round(sma_data['50'].iloc[-1], 2),
        "SMA150": np.round(sma_data['150'].iloc[-1], 2),
        "SMA200": np.round(sma_data['200'].iloc[-1], 2),
        "Price vs SMA150": price_vs_sma['150'].iloc[-1],
        "SMA50 vs SMA200": sma_relation['signal'].iloc[-1],
        "RSI(14)": np.round(rsi_short.iloc[-1], 2),
        "RSI(50)": np.round(rsi_long.iloc[-1], 2),
        "ADX(14)": np.round(adx_short.iloc[-1], 2),
        "ADX(50)": np.round(adx_long.iloc[-1], 2),
        "MACD(12, 26, 9)": (
            f"{macd_short['macd_level'].iloc[-1]} ({np.round(macd_short['macd_value'].iloc[-1], 2)}) "
            f"{macd_short_bullish} {macd_short_bearish} {macd_short_near_zero}"
        ),
        "MACD(50, 20, 9)": (
            f"{macd_long['macd_level'].iloc[-1]} ({np.round(macd_long['macd_value'].iloc[-1], 2)}) "
            f"{macd_long_bullish} {macd_long_bearish} {macd_long_near_zero}"
        ),
        "Bollinger Band(20)": bollinger['position'].iloc[-1],
        "Bollinger Upper Band": bollinger['upper_band'].iloc[-1],
        "Bollinger Lower Band": bollinger['lower_band'].iloc[-1],
        "Fibonacci(90)": fibonacci.iloc[-1],
        "Aroon(14) Up, Down": f"{np.round(aroon_up.iloc[-1], 2)}, {np.round(aroon_down.iloc[-1], 2)}",
        "Stochastic %K, %D": f"{np.round(stochastic_data['%K'].iloc[-1], 2)}, {np.round(stochastic_data['%D'].iloc[-1], 2)}",
    }

    return pd.DataFrame(personal_retracement, index=[0])

def main():
    """
    Main function to run the analysis for multiple tickers and save results.
    """
    config = load_config()
    tickers = config["TICKER_LIST"].split(',')

    # Start spinner in a thread
    spin_thread = threading.Thread(target=spinner)
    spin_thread.start()

    results = pd.DataFrame()
    for ticker in tickers:
        df = get_single_ticker_data(ticker)
        results = pd.concat([results, df], ignore_index=True)

    markdown_table = results.to_markdown(index=False)
    today = datetime.now().strftime('%Y%m%d')
    write_to_file(markdown_table, f"Overall_{today}.md")


    # Stop spinner
    spin_thread.stop = True
    spin_thread.join()

if __name__ == "__main__":
    main()