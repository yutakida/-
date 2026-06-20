import json
from mcp.server.fastmcp import FastMCP
from . import data, indicators, patterns, backtest

mcp = FastMCP("trading-mcp")


@mcp.tool()
def get_historical_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
) -> str:
    """
    銘柄の過去OHLCVデータを取得する。

    Args:
        symbol: 銘柄コード (例: AAPL, BTC-USD, 7203.T, ^N225)
        period: 期間 (1d/5d/1mo/3mo/6mo/1y/2y/5y/max)
        interval: 足種 (1m/5m/15m/30m/1h/1d/1wk/1mo)
    """
    df = data.fetch_ohlcv(symbol, period, interval)
    recent = df[["Open", "High", "Low", "Close", "Volume"]].tail(20).round(4)
    result = {
        "symbol": symbol,
        "period": period,
        "interval": interval,
        "total_bars": len(df),
        "start": str(df.index[0].date() if hasattr(df.index[0], "date") else df.index[0]),
        "end": str(df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]),
        "latest_close": round(float(df["Close"].iloc[-1]), 4),
        "recent_20_bars": recent.to_dict(orient="index"),
    }
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool()
def calculate_indicator(
    symbol: str,
    indicator: str,
    period: str = "1y",
    interval: str = "1d",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    std: float = 2.0,
    k: int = 14,
    d: int = 3,
    length: int = 14,
) -> str:
    """
    テクニカルインジケータを計算して返す。

    Args:
        symbol: 銘柄コード
        indicator: インジケータ名 (sma/ema/rsi/macd/bbands/atr/stoch/adx/vwap)
        period: データ取得期間
        interval: 足種
        fast: MACD短期期間 (MACD用)
        slow: MACD長期期間 (MACD用)
        signal: MACDシグナル期間 (MACD用)
        std: 標準偏差倍率 (BBands用)
        k: %K期間 (Stoch用)
        d: %D期間 (Stoch用)
        length: 汎用期間 (SMA/EMA/RSI/ATR/ADX用)
    """
    df = data.fetch_ohlcv(symbol, period, interval)
    params = {
        "period": length,
        "fast": fast, "slow": slow, "signal": signal,
        "std": std, "k": k, "d": d,
    }
    result = indicators.calculate(df, indicator, **params)
    result["symbol"] = symbol
    result["interval"] = interval
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool()
def detect_patterns(
    symbol: str,
    pattern_type: str = "candlestick",
    period: str = "1y",
    interval: str = "1d",
    short_ma: int = 50,
    long_ma: int = 200,
) -> str:
    """
    チャートパターン・ローソク足パターンを検出する。

    Args:
        symbol: 銘柄コード
        pattern_type: パターン種別 (candlestick/ma_crossover)
        period: データ取得期間
        interval: 足種
        short_ma: 短期MA期間 (ma_crossover用)
        long_ma: 長期MA期間 (ma_crossover用)
    """
    df = data.fetch_ohlcv(symbol, period, interval)

    if pattern_type == "candlestick":
        result = patterns.detect_candlestick(df)
    elif pattern_type == "ma_crossover":
        result = patterns.detect_ma_crossover(df, short=short_ma, long=long_ma)
    else:
        raise ValueError(f"未対応のパターン: {pattern_type}。対応: candlestick, ma_crossover")

    result["symbol"] = symbol
    result["pattern_type"] = pattern_type
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool()
def find_support_resistance(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    window: int = 10,
    levels: int = 5,
) -> str:
    """
    サポート・レジスタンスラインを検出する。

    Args:
        symbol: 銘柄コード
        period: データ取得期間
        interval: 足種
        window: ピーク検出ウィンドウ幅
        levels: 返すレベル数
    """
    df = data.fetch_ohlcv(symbol, period, interval)
    result = patterns.find_support_resistance(df, window=window, levels=levels)
    result["symbol"] = symbol
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool()
def backtest_strategy(
    symbol: str,
    strategy: str = "ma_crossover",
    period: str = "2y",
    interval: str = "1d",
    initial_capital: float = 1_000_000,
    short_period: int = 20,
    long_period: int = 50,
    rsi_period: int = 14,
    oversold: float = 30,
    overbought: float = 70,
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> str:
    """
    戦略をバックテストして損益・ドローダウン・シャープレシオ等を返す。

    Args:
        symbol: 銘柄コード
        strategy: 戦略名 (ma_crossover/rsi/bbands)
        period: バックテスト期間
        interval: 足種
        initial_capital: 初期資金
        short_period: 短期期間 (ma_crossover用)
        long_period: 長期期間 (ma_crossover用)
        rsi_period: RSI期間 (rsi用)
        oversold: RSI売られすぎ閾値 (rsi用)
        overbought: RSI買われすぎ閾値 (rsi用)
        bb_period: BBands期間 (bbands用)
        bb_std: BBands標準偏差倍率 (bbands用)
    """
    df = data.fetch_ohlcv(symbol, period, interval)

    if strategy == "ma_crossover":
        signals = backtest.ma_crossover_signals(df, short=short_period, long=long_period)
    elif strategy == "rsi":
        signals = backtest.rsi_signals(df, period=rsi_period, oversold=oversold, overbought=overbought)
    elif strategy == "bbands":
        signals = backtest.bbands_signals(df, period=bb_period, std=bb_std)
    else:
        raise ValueError(f"未対応の戦略: {strategy}。対応: ma_crossover, rsi, bbands")

    result = backtest.run(df, signals, initial_capital=initial_capital)
    result["symbol"] = symbol
    result["strategy"] = strategy
    result["period"] = period
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool()
def get_ticker_info(symbol: str) -> str:
    """
    銘柄の基本情報（企業名・セクター・時価総額・52週高安値等）を取得する。

    Args:
        symbol: 銘柄コード
    """
    info = data.get_ticker_info(symbol)
    info["symbol"] = symbol
    return json.dumps(info, ensure_ascii=False, default=str)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
