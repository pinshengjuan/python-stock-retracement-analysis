# Stock Technical Indicator Analysis

This project analyzes stock data and generates a comprehensive technical indicator report for selected tickers. It fetches historical price and volume data, computes various technical indicators, and outputs results in a markdown table.

## Features

- Fetches historical stock data using Yahoo Finance (`yfinance`)
- Calculates technical indicators:
  - Simple Moving Averages (SMA)
  - Relative Strength Index (RSI)
  - Average Directional Index (ADX)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Fibonacci Retracement Levels
  - Aroon Indicator
  - Stochastic Oscillator
  - Volume metrics
- Generates markdown reports with all computed indicators
- Configurable via `.env` file

## Project Structure

- `app.py`: Main script to run analysis and generate report
- `indicator_calculation/`: Technical indicator calculation modules
- `signal_helper_utilities/`: Signal logic, data fetching, reporting, and utilities
- `.env`: Configuration for tickers and indicator periods
- `requirements.txt`: Python dependencies

## Installation

1. Clone the repository.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Create a `.env` file in the project directory with your stock tickers, e.g.:
    ```
    TICKER_LIST=AAPL,MSFT,GOOGL
    ```

2. Run the script:
    ```sh
    python app.py
    ```
The output markdown report will be saved in the data/ directory.

## Configuration
Edit the .env file to set:

- TICKER_LIST: Comma-separated tickers (e.g., 'AAPL,ARM,NVDA')
- Indicator periods (e.g., DAYS, RSI_PERIOD_SHORT, etc.)

## License

MIT License