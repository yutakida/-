import pandas as pd
import pandas_ta as ta


def calculate(df: pd.DataFrame, name: str, **params) -> dict:
    name = name.lower()

    if name == "sma":
        period = params.get("period", 20)
        result = ta.sma(df["Close"], length=period)
        return _series_to_dict(result, f"SMA_{period}")

    if name == "ema":
        period = params.get("period", 20)
        result = ta.ema(df["Close"], length=period)
        return _series_to_dict(result, f"EMA_{period}")

    if name == "rsi":
        period = params.get("period", 14)
        result = ta.rsi(df["Close"], length=period)
        return _series_to_dict(result, f"RSI_{period}")

    if name == "macd":
        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        signal = params.get("signal", 9)
        result = ta.macd(df["Close"], fast=fast, slow=slow, signal=signal)
        return _df_to_dict(result)

    if name == "bbands":
        period = params.get("period", 20)
        std = params.get("std", 2.0)
        result = ta.bbands(df["Close"], length=period, std=std)
        return _df_to_dict(result)

    if name == "atr":
        period = params.get("period", 14)
        result = ta.atr(df["High"], df["Low"], df["Close"], length=period)
        return _series_to_dict(result, f"ATR_{period}")

    if name == "stoch":
        k = params.get("k", 14)
        d = params.get("d", 3)
        result = ta.stoch(df["High"], df["Low"], df["Close"], k=k, d=d)
        return _df_to_dict(result)

    if name == "adx":
        period = params.get("period", 14)
        result = ta.adx(df["High"], df["Low"], df["Close"], length=period)
        return _df_to_dict(result)

    if name == "vwap":
        result = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
        return _series_to_dict(result, "VWAP")

    raise ValueError(f"未対応のインジケータ: {name}。対応: sma, ema, rsi, macd, bbands, atr, stoch, adx, vwap")


def _series_to_dict(series: pd.Series, label: str) -> dict:
    recent = series.dropna().tail(30)
    return {
        "indicator": label,
        "latest": round(float(recent.iloc[-1]), 4) if not recent.empty else None,
        "data": {str(k.date() if hasattr(k, "date") else k): round(float(v), 4)
                 for k, v in recent.items()},
    }


def _df_to_dict(df: pd.DataFrame) -> dict:
    recent = df.dropna().tail(30)
    return {
        "columns": list(recent.columns),
        "latest": {col: round(float(recent[col].iloc[-1]), 4) for col in recent.columns}
        if not recent.empty else {},
        "data": recent.round(4).to_dict(orient="index",
                                         into=dict) if not recent.empty else {},
    }
