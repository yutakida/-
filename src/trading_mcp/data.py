import yfinance as yf
import pandas as pd


def fetch_ohlcv(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"データが取得できませんでした: {symbol}")
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    return df


def get_ticker_info(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)
    info = ticker.info
    keys = [
        "longName", "shortName", "sector", "industry", "country",
        "marketCap", "currentPrice", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        "averageVolume", "currency", "exchange",
    ]
    return {k: info.get(k) for k in keys if info.get(k) is not None}
