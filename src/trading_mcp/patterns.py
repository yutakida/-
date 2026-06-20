import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Any


def detect_candlestick(df: pd.DataFrame) -> dict:
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    body = (close - open_).abs()
    range_ = high - low
    upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low

    doji = (range_ > 0) & (body / range_ < 0.05)

    hammer = (
        (lower_shadow > 2 * body) &
        (upper_shadow < body * 0.5) &
        (body > 0)
    )

    shooting_star = (
        (upper_shadow > 2 * body) &
        (lower_shadow < body * 0.5) &
        (body > 0)
    )

    bull_engulf = (
        (close.shift(1) < open_.shift(1)) &
        (close > open_) &
        (open_ < close.shift(1)) &
        (close > open_.shift(1))
    )

    bear_engulf = (
        (close.shift(1) > open_.shift(1)) &
        (close < open_) &
        (open_ > close.shift(1)) &
        (close < open_.shift(1))
    )

    patterns = pd.DataFrame({
        "doji": doji,
        "hammer": hammer,
        "shooting_star": shooting_star,
        "bullish_engulfing": bull_engulf,
        "bearish_engulfing": bear_engulf,
    })

    detected: list[dict[str, Any]] = []
    for pattern_name in patterns.columns:
        hits = patterns[pattern_name][patterns[pattern_name]].index
        for dt in hits[-10:]:
            detected.append({
                "date": str(dt.date() if hasattr(dt, "date") else dt),
                "pattern": pattern_name,
                "close": round(float(close[dt]), 4),
            })

    detected.sort(key=lambda x: x["date"], reverse=True)
    return {"detected": detected[:20]}


def detect_ma_crossover(df: pd.DataFrame, short: int = 50, long: int = 200) -> dict:
    sma_short = ta.sma(df["Close"], length=short)
    sma_long = ta.sma(df["Close"], length=long)

    golden = (sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))
    death = (sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))

    events: list[dict[str, Any]] = []
    for dt in golden[golden].index:
        events.append({
            "date": str(dt.date() if hasattr(dt, "date") else dt),
            "type": "golden_cross",
            "close": round(float(df["Close"][dt]), 4),
        })
    for dt in death[death].index:
        events.append({
            "date": str(dt.date() if hasattr(dt, "date") else dt),
            "type": "death_cross",
            "close": round(float(df["Close"][dt]), 4),
        })

    events.sort(key=lambda x: x["date"], reverse=True)

    current_trend = "bullish" if (
        not sma_short.dropna().empty and not sma_long.dropna().empty and
        sma_short.dropna().iloc[-1] > sma_long.dropna().iloc[-1]
    ) else "bearish"

    return {
        "short_ma": short,
        "long_ma": long,
        "current_trend": current_trend,
        "recent_events": events[:10],
    }


def find_support_resistance(df: pd.DataFrame, window: int = 10, levels: int = 5) -> dict:
    highs = df["High"]
    lows = df["Low"]

    local_max_mask = (highs == highs.rolling(window=window * 2 + 1, center=True).max())
    local_min_mask = (lows == lows.rolling(window=window * 2 + 1, center=True).min())

    resistance_vals = highs[local_max_mask].dropna().values
    support_vals = lows[local_min_mask].dropna().values

    resistance = _cluster_levels(resistance_vals, levels)
    support = _cluster_levels(support_vals, levels)

    current_price = float(df["Close"].iloc[-1])

    return {
        "current_price": round(current_price, 4),
        "resistance": [round(r, 4) for r in sorted(resistance) if r > current_price][:levels],
        "support": [round(s, 4) for s in sorted(resistance, reverse=True) if s < current_price][:0]
                   + [round(s, 4) for s in sorted(support, reverse=True) if s < current_price][:levels],
    }


def _cluster_levels(values: np.ndarray, n: int, tol_pct: float = 0.02) -> list[float]:
    if len(values) == 0:
        return []
    clusters: list[float] = []
    for v in sorted(values, reverse=True):
        if not any(abs(v - c) / c < tol_pct for c in clusters):
            clusters.append(v)
        if len(clusters) >= n:
            break
    return clusters
