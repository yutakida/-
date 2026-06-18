"""ゴールド（GC=F）のチャートパターン出現率検証スクリプト"""
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ---------- インジケータ（pandas-ta不要の手動実装） ----------

def sma(series, n):
    return series.rolling(n).mean()

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def macd(series, fast=12, slow=26, signal=9):
    m = ema(series, fast) - ema(series, slow)
    s = ema(m, signal)
    return m, s

def bbands(series, n=20, std=2.0):
    mid = sma(series, n)
    sd = series.rolling(n).std()
    return mid + std * sd, mid, mid - std * sd


# ---------- パターン検出 ----------

def detect_patterns(df):
    op, hi, lo, cl = df["Open"], df["High"], df["Low"], df["Close"]
    body = (cl - op).abs()
    rng = hi - lo
    upper_wick = hi - pd.concat([op, cl], axis=1).max(axis=1)
    lower_wick = pd.concat([op, cl], axis=1).min(axis=1) - lo

    results = {}

    # ローソク足パターン
    results["doji"] = (rng > 0) & (body / rng < 0.05)

    results["hammer"] = (
        (lower_wick > 2 * body) & (upper_wick < body * 0.5) & (body > 0)
    )
    results["shooting_star"] = (
        (upper_wick > 2 * body) & (lower_wick < body * 0.5) & (body > 0)
    )
    results["bullish_engulfing"] = (
        (cl.shift(1) < op.shift(1)) & (cl > op) &
        (op < cl.shift(1)) & (cl > op.shift(1))
    )
    results["bearish_engulfing"] = (
        (cl.shift(1) > op.shift(1)) & (cl < op) &
        (op > cl.shift(1)) & (cl < op.shift(1))
    )

    # MAクロス
    sma50 = sma(cl, 50)
    sma200 = sma(cl, 200)
    results["golden_cross"] = (sma50 > sma200) & (sma50.shift(1) <= sma200.shift(1))
    results["death_cross"]  = (sma50 < sma200) & (sma50.shift(1) >= sma200.shift(1))

    # RSIダイバージェンス代替：RSI極値
    r = rsi(cl)
    results["rsi_oversold"]   = r < 30
    results["rsi_overbought"] = r > 70

    # BBandsブレイク
    upper, _, lower = bbands(cl)
    results["bb_breakout_up"]   = cl > upper
    results["bb_breakout_down"] = cl < lower

    # MACD
    m, s = macd(cl)
    results["macd_bullish_cross"] = (m > s) & (m.shift(1) <= s.shift(1))
    results["macd_bearish_cross"] = (m < s) & (m.shift(1) >= s.shift(1))

    return pd.DataFrame(results)


# ---------- 出現率 + 事後リターン ----------

def analyze_pattern_stats(df, pattern_df, forward_days=5):
    cl = df["Close"]
    fwd_ret = cl.shift(-forward_days) / cl - 1  # n日後リターン

    rows = []
    for pat in pattern_df.columns:
        mask = pattern_df[pat].fillna(False)
        count = int(mask.sum())
        total = len(mask.dropna())
        rate  = count / total * 100 if total > 0 else 0

        if count > 0:
            returns = fwd_ret[mask].dropna()
            win_rate   = (returns > 0).mean() * 100
            avg_return = returns.mean() * 100
            last_date  = df.index[mask].max()
            last_date  = str(last_date.date() if hasattr(last_date, "date") else last_date)
        else:
            win_rate = avg_return = float("nan")
            last_date = "-"

        rows.append({
            "パターン": pat,
            "出現回数": count,
            "総バー数": total,
            "出現率(%)": round(rate, 2),
            f"{forward_days}日後勝率(%)": round(win_rate, 1) if not np.isnan(win_rate) else "-",
            f"{forward_days}日後平均騰落(%)": round(avg_return, 2) if not np.isnan(avg_return) else "-",
            "最終出現日": last_date,
        })

    return pd.DataFrame(rows).sort_values("出現率(%)", ascending=False)


# ---------- メイン ----------

def main():
    print("=" * 60)
    print("ゴールド（GC=F）チャートパターン出現率検証")
    print("=" * 60)

    symbols = {"ゴールド先物(GC=F)": "GC=F", "金ETF(GLD)": "GLD"}
    for label, sym in symbols.items():
        print(f"\n▼ {label}  ／  期間: 過去5年 日足\n")
        try:
            df = yf.Ticker(sym).history(period="5y", interval="1d")
        except Exception as e:
            print(f"  データ取得エラー: {e}")
            continue

        if df.empty:
            print("  データなし")
            continue

        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        print(f"  取得バー数: {len(df)}  ({df.index[0].date()} 〜 {df.index[-1].date()})")
        print(f"  現在値: {df['Close'].iloc[-1]:.2f}")
        print()

        pats = detect_patterns(df)
        stats = analyze_pattern_stats(df, pats, forward_days=5)

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        pd.set_option("display.float_format", "{:.2f}".format)
        print(stats.to_string(index=False))
        print()

        # 直近のシグナル
        print("  ▼ 直近20バーで出現したパターン")
        recent = pats.tail(20)
        found = []
        for pat in recent.columns:
            hits = recent[pat][recent[pat]].index
            for dt in hits:
                found.append((str(dt.date() if hasattr(dt, "date") else dt), pat,
                              round(float(df.loc[dt, "Close"]), 2)))
        if found:
            found.sort(key=lambda x: x[0], reverse=True)
            for date, pat, price in found:
                print(f"    {date}  {pat:<25}  終値: {price}")
        else:
            print("    （なし）")

    print("\n" + "=" * 60)
    print("完了")


if __name__ == "__main__":
    main()
