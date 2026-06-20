"""
ゴールドのチャートパターン出現率検証（合成データ版）
※ 本番ではyfinanceで GC=F / GLD のデータを取得して実行してください
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# ---------- ゴールド統計特性に基づく合成OHLCV生成 ----------

def generate_gold_ohlcv(n=1260, start_price=1800.0, annual_vol=0.15, annual_drift=0.07):
    """幾何ブラウン運動 + ジャンプでゴールドらしい値動きを生成"""
    dt = 1 / 252
    daily_vol = annual_vol * np.sqrt(dt)
    daily_drift = (annual_drift - 0.5 * annual_vol ** 2) * dt

    jumps = np.random.binomial(1, 0.02, n) * np.random.normal(0, 0.015, n)
    returns = daily_drift + daily_vol * np.random.randn(n) + jumps

    close = np.zeros(n)
    close[0] = start_price
    for i in range(1, n):
        close[i] = close[i - 1] * np.exp(returns[i])

    intraday_range = np.abs(np.random.normal(0, daily_vol * 0.6, n))
    open_ = close * (1 + np.random.uniform(-0.5, 0.5, n) * intraday_range)
    high   = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, intraday_range * 0.5)))
    low    = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, intraday_range * 0.5)))
    volume = np.random.lognormal(mean=12.0, sigma=0.4, size=n).astype(int)

    dates = pd.date_range(end=pd.Timestamp("2026-06-18"), periods=n, freq="B")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                          "Close": close, "Volume": volume}, index=dates)


# ---------- インジケータ ----------

def sma(s, n): return s.rolling(n).mean()
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def macd_cross(s, fast=12, slow=26, signal=9):
    m = ema(s, fast) - ema(s, slow)
    sig = ema(m, signal)
    return (m > sig) & (m.shift(1) <= sig.shift(1)), (m < sig) & (m.shift(1) >= sig.shift(1))

def bbands(s, n=20, std=2.0):
    mid = sma(s, n); sd = s.rolling(n).std()
    return mid + std * sd, mid - std * sd


# ---------- パターン検出 ----------

def detect_all(df):
    op, hi, lo, cl = df["Open"], df["High"], df["Low"], df["Close"]
    body = (cl - op).abs()
    rng  = hi - lo
    upper_wick = hi - pd.concat([op, cl], axis=1).max(axis=1)
    lower_wick = pd.concat([op, cl], axis=1).min(axis=1) - lo

    r   = rsi(cl)
    ub, lb = bbands(cl)
    mc_bull, mc_bear = macd_cross(cl)

    s50,  s200 = sma(cl, 50),  sma(cl, 200)
    e20 = ema(cl, 20)

    return pd.DataFrame({
        # ローソク足
        "doji":              (rng > 0) & (body / rng < 0.05),
        "hammer":            (lower_wick > 2*body) & (upper_wick < body*0.5) & (body > 0),
        "shooting_star":     (upper_wick > 2*body) & (lower_wick < body*0.5) & (body > 0),
        "bullish_engulfing": (cl.shift(1)<op.shift(1)) & (cl>op) & (op<cl.shift(1)) & (cl>op.shift(1)),
        "bearish_engulfing": (cl.shift(1)>op.shift(1)) & (cl<op) & (op>cl.shift(1)) & (cl<op.shift(1)),
        # MAクロス
        "golden_cross":      (s50 > s200) & (s50.shift(1) <= s200.shift(1)),
        "death_cross":       (s50 < s200) & (s50.shift(1) >= s200.shift(1)),
        "ema20_cross_up":    (cl > e20)   & (cl.shift(1) <= e20.shift(1)),
        "ema20_cross_down":  (cl < e20)   & (cl.shift(1) >= e20.shift(1)),
        # RSI
        "rsi_oversold":      r < 30,
        "rsi_overbought":    r > 70,
        # BBands
        "bb_breakout_up":    cl > ub,
        "bb_breakout_down":  cl < lb,
        # MACD
        "macd_bull_cross":   mc_bull,
        "macd_bear_cross":   mc_bear,
    })


# ---------- 統計 ----------

def stats(df, pat_df, fwd=5):
    cl = df["Close"]
    fwd_ret = cl.shift(-fwd) / cl - 1

    rows = []
    total = len(pat_df)
    for pat in pat_df.columns:
        mask  = pat_df[pat].fillna(False)
        count = int(mask.sum())
        rate  = count / total * 100

        rets  = fwd_ret[mask].dropna()
        win   = (rets > 0).mean() * 100 if count > 0 else float("nan")
        avg   = rets.mean() * 100        if count > 0 else float("nan")
        last  = str(df.index[mask].max().date()) if count > 0 else "-"

        rows.append({
            "パターン名":         pat,
            "出現回数":           count,
            "出現率(%)":          round(rate, 2),
            f"{fwd}日後勝率(%)":  f"{win:.1f}" if not np.isnan(win) else "-",
            f"{fwd}日後平均騰落": f"{avg:+.2f}%" if not np.isnan(avg) else "-",
            "最終出現日":         last,
        })

    return pd.DataFrame(rows).sort_values("出現率(%)", ascending=False)


# ---------- メイン ----------

def main():
    print("=" * 70)
    print("ゴールド チャートパターン出現率・事後リターン検証")
    print("データ: 過去5年 日足（合成データ ─ GBM + ジャンプ、年率vol=15%）")
    print("※ 本番では yfinance GC=F / GLD に差し替えてください")
    print("=" * 70)

    df = generate_gold_ohlcv(n=1260, start_price=1800.0)
    print(f"\n期間: {df.index[0].date()} 〜 {df.index[-1].date()}  ({len(df)} 本)")
    print(f"始値: {df['Close'].iloc[0]:.0f}  →  終値: {df['Close'].iloc[-1]:.0f}"
          f"  (騰落: {(df['Close'].iloc[-1]/df['Close'].iloc[0]-1)*100:+.1f}%)\n")

    pat_df  = detect_all(df)
    result  = stats(df, pat_df, fwd=5)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 110)
    print(result.to_string(index=False))

    # 直近30日のシグナル
    print("\n" + "-" * 70)
    print("直近30日に出現したパターン")
    print("-" * 70)
    recent = pat_df.tail(30)
    found  = []
    for pat in recent.columns:
        hits = recent[pat][recent[pat]].index
        for dt in hits:
            found.append((str(dt.date()), pat, round(float(df.loc[dt, "Close"]), 0)))
    if found:
        found.sort(key=lambda x: x[0], reverse=True)
        for date, pat, price in found:
            print(f"  {date}  {pat:<25}  終値: {price:,.0f}")
    else:
        print("  （なし）")

    # 高精度パターンTOP3
    valid = result[result["出現回数"] >= 10].copy()
    valid["勝率_num"] = pd.to_numeric(
        valid[f"5日後勝率(%)"].str.replace("%", "").replace("-", float("nan")),
        errors="coerce"
    )
    print("\n" + "-" * 70)
    print("5日後勝率 上位3パターン（出現10回以上）")
    print("-" * 70)
    top = valid.nlargest(3, "勝率_num")
    for _, row in top.iterrows():
        print(f"  {row['パターン名']:<25} 勝率: {row['5日後勝率(%)']:>6}  "
              f"平均騰落: {row['5日後平均騰落']:>8}  出現率: {row['出現率(%)']:.2f}%")

    print("\n" + "=" * 70)
    print("実運用のポイント:")
    print("  1. golden_cross / death_cross は出現頻度が低い分、信頼度が高い傾向")
    print("  2. rsi_oversold + hammer の重複出現 → 強い買いシグナル候補")
    print("  3. bb_breakout_up 後は反転リスクも検討（平均回帰）")


if __name__ == "__main__":
    main()
