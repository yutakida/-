import pandas as pd
import numpy as np
import pandas_ta as ta


def run(df: pd.DataFrame, signals: pd.Series, initial_capital: float = 1_000_000) -> dict:
    """
    signals: +1=買い, -1=売り, 0=ポジションなし (各日の終値で執行)
    """
    position = signals.shift(1).fillna(0)
    daily_returns = df["Close"].pct_change().fillna(0)
    strategy_returns = position * daily_returns

    equity = (1 + strategy_returns).cumprod() * initial_capital
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    trades = signals.diff().ne(0) & signals.ne(0)
    trade_returns = strategy_returns[trades.shift(-1).fillna(False)]

    total_return = float((equity.iloc[-1] / initial_capital - 1) * 100)
    max_drawdown = float(drawdown.min() * 100)
    num_trades = int(trades.sum())
    winning = (trade_returns > 0).sum()
    win_rate = float(winning / num_trades * 100) if num_trades > 0 else 0.0

    ann_factor = 252 if len(df) > 60 else 52
    sharpe = float(
        strategy_returns.mean() / strategy_returns.std() * np.sqrt(ann_factor)
    ) if strategy_returns.std() > 0 else 0.0

    buy_hold_return = float((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100)

    return {
        "total_return_pct": round(total_return, 2),
        "buy_hold_return_pct": round(buy_hold_return, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe, 3),
        "num_trades": num_trades,
        "win_rate_pct": round(win_rate, 2),
        "final_capital": round(float(equity.iloc[-1]), 0),
    }


def ma_crossover_signals(df: pd.DataFrame, short: int = 20, long: int = 50) -> pd.Series:
    sma_s = ta.sma(df["Close"], length=short)
    sma_l = ta.sma(df["Close"], length=long)
    signals = pd.Series(0, index=df.index)
    signals[sma_s > sma_l] = 1
    signals[sma_s < sma_l] = -1
    return signals.fillna(0)


def rsi_signals(
    df: pd.DataFrame,
    period: int = 14,
    oversold: float = 30,
    overbought: float = 70,
) -> pd.Series:
    rsi = ta.rsi(df["Close"], length=period)
    signals = pd.Series(0, index=df.index)
    signals[rsi < oversold] = 1
    signals[rsi > overbought] = -1
    return signals.fillna(0)


def bbands_signals(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.Series:
    bb = ta.bbands(df["Close"], length=period, std=std)
    lower_col = [c for c in bb.columns if "BBL" in c][0]
    upper_col = [c for c in bb.columns if "BBU" in c][0]
    signals = pd.Series(0, index=df.index)
    signals[df["Close"] < bb[lower_col]] = 1
    signals[df["Close"] > bb[upper_col]] = -1
    return signals.fillna(0)
