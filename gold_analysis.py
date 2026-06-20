"""
ゴールドチャート取得 & パターン分析
- データソース: GC=F（ゴールド先物）または GLD（金ETF）
- ローカル実行: pip install yfinance pandas numpy && python3 gold_analysis.py
"""
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────
# データ取得（yfinance / 合成データフォールバック）
# ──────────────────────────────────────────

def fetch_gold(symbol="GC=F", period="5y", interval="1d") -> pd.DataFrame:
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df.empty:
            raise ValueError("データなし")
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        print(f"✓ {symbol} データ取得完了 ({len(df)} 本)")
        return df
    except Exception as e:
        print(f"⚠  yfinance 取得失敗 ({e}) → 合成データで代替")
        return _synthetic_gold()


def _synthetic_gold(n=1260, start=1800.0, drift=0.07, vol=0.15) -> pd.DataFrame:
    """ゴールドの実際の統計特性（年率 vol≈15%、長期上昇）を再現"""
    np.random.seed(42)
    dt = 1 / 252
    ret = (drift - .5*vol**2)*dt + vol*np.sqrt(dt)*np.random.randn(n)
    ret += np.random.binomial(1, .02, n) * np.random.normal(0, .015, n)  # ジャンプ
    cl  = start * np.exp(np.cumsum(ret))
    rng = np.abs(np.random.normal(0, vol*np.sqrt(dt)*.6, n))
    op  = cl * (1 + np.random.uniform(-.5, .5, n)*rng)
    hi  = np.maximum(op, cl) * (1 + np.abs(np.random.normal(0, rng*.5)))
    lo  = np.minimum(op, cl) * (1 - np.abs(np.random.normal(0, rng*.5)))
    vol_ = np.random.lognormal(12., .4, n).astype(int)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="B")
    return pd.DataFrame({"Open":op,"High":hi,"Low":lo,"Close":cl,"Volume":vol_}, index=dates)


# ──────────────────────────────────────────
# インジケータ
# ──────────────────────────────────────────

def sma(s,n): return s.rolling(n).mean()
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); g=d.clip(lower=0).rolling(n).mean(); l=(-d.clip(upper=0)).rolling(n).mean()
    return 100-100/(1+g/l.replace(0,np.nan))
def bbands(s,n=20,k=2.):
    m=sma(s,n); sd=s.rolling(n).std(); return m+k*sd, m, m-k*sd
def macd(s,f=12,sl=26,sg=9):
    m=ema(s,f)-ema(s,sl); sig=ema(m,sg); return m,sig
def atr(hi,lo,cl,n=14):
    tr=pd.concat([hi-lo,(cl.shift()-hi).abs(),(cl.shift()-lo).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()


# ──────────────────────────────────────────
# パターン検出
# ──────────────────────────────────────────

def detect(df):
    op,hi,lo,cl = df["Open"],df["High"],df["Low"],df["Close"]
    body = (cl-op).abs(); rng = hi-lo
    hi_oc = pd.concat([op,cl],axis=1).max(axis=1)
    lo_oc = pd.concat([op,cl],axis=1).min(axis=1)
    uw = hi-hi_oc; lw = lo_oc-lo

    s20,s50,s200 = sma(cl,20),sma(cl,50),sma(cl,200)
    r = rsi(cl); ub,mb,lb = bbands(cl); m,sig = macd(cl)

    return pd.DataFrame({
        # ローソク足
        "doji":              (rng>0)&(body/rng<.05),
        "hammer":            (lw>2*body)&(uw<body*.5)&(body>0),
        "shooting_star":     (uw>2*body)&(lw<body*.5)&(body>0),
        "bullish_engulfing": (cl.shift()<op.shift())&(cl>op)&(op<cl.shift())&(cl>op.shift()),
        "bearish_engulfing": (cl.shift()>op.shift())&(cl<op)&(op>cl.shift())&(cl<op.shift()),
        # MA
        "golden_cross":      (s50>s200)&(s50.shift()<=s200.shift()),
        "death_cross":       (s50<s200)&(s50.shift()>=s200.shift()),
        "price_above_sma20": cl>s20,
        # モメンタム
        "rsi_oversold":      r<30,
        "rsi_overbought":    r>70,
        "macd_bull":         (m>sig)&(m.shift()<=sig.shift()),
        "macd_bear":         (m<sig)&(m.shift()>=sig.shift()),
        # ボラ
        "bb_squeeze":        (ub-lb)/mb < (ub-lb).rolling(20).mean()/mb*0.7,
        "bb_break_up":       cl>ub,
        "bb_break_down":     cl<lb,
    })


# ──────────────────────────────────────────
# 統計
# ──────────────────────────────────────────

def pattern_stats(df, pat_df, fwd_days=5):
    cl = df["Close"]
    fwd = cl.shift(-fwd_days)/cl-1
    total = len(pat_df)
    rows=[]
    for p in pat_df.columns:
        mask = pat_df[p].fillna(False)
        cnt  = int(mask.sum())
        rets = fwd[mask].dropna()
        rows.append({
            "パターン":        p,
            "出現回数":        cnt,
            "出現率%":         round(cnt/total*100,2),
            f"{fwd_days}日後勝率%": round((rets>0).mean()*100,1) if cnt>0 else None,
            f"{fwd_days}日後平均%": round(rets.mean()*100,3)    if cnt>0 else None,
            "最終出現":        str(df.index[mask].max().date()) if cnt>0 else "-",
        })
    return pd.DataFrame(rows).sort_values("出現率%",ascending=False)


# ──────────────────────────────────────────
# メイン
# ──────────────────────────────────────────

def main():
    SYMBOL   = "GC=F"   # ゴールド先物（GLD = ETFも可）
    PERIOD   = "5y"
    INTERVAL = "1d"
    FWD      = 5        # 事後リターン確認日数

    print("═"*65)
    print(f" ゴールド チャートパターン分析  [{SYMBOL}  {PERIOD} {INTERVAL}]")
    print("═"*65)

    df = fetch_gold(SYMBOL, PERIOD, INTERVAL)
    cl = df["Close"]
    s50, s200  = sma(cl,50), sma(cl,200)
    r          = rsi(cl)
    ub,_,lb    = bbands(cl)
    at         = atr(df["High"],df["Low"],cl)
    m, sig_m   = macd(cl)

    print(f"\n期間  : {df.index[0].date()} 〜 {df.index[-1].date()}  ({len(df)} 本)")
    print(f"終値  : ${cl.iloc[-1]:,.2f}")
    print(f"SMA50 : ${s50.iloc[-1]:,.2f}  SMA200: ${s200.iloc[-1]:,.2f}")
    print(f"RSI14 : {r.iloc[-1]:.1f}")
    print(f"ATR14 : ${at.iloc[-1]:,.2f}  ({at.iloc[-1]/cl.iloc[-1]*100:.2f}%)")
    trend = "強気" if s50.iloc[-1] > s200.iloc[-1] else "弱気"
    print(f"トレンド: {trend}  (SMA50 {'>' if trend=='強気' else '<'} SMA200)\n")

    pat_df = detect(df)
    stats  = pattern_stats(df, pat_df, FWD)

    print("─"*65)
    print("【パターン出現率・事後リターン一覧】")
    print("─"*65)
    pd.set_option("display.width",110)
    pd.set_option("display.max_columns",None)
    print(stats.fillna("-").to_string(index=False))

    print("\n" + "─"*65)
    print("【直近30日のシグナル】")
    print("─"*65)
    recent = pat_df.tail(30)
    hits = [(str(df.index[df.index.get_loc(dt)].date()), pat,
             round(float(cl.loc[dt]),2))
            for pat in recent.columns
            for dt in recent.index[recent[pat].fillna(False)]]
    hits.sort(key=lambda x:x[0], reverse=True)
    if hits:
        for d,p,pr in hits:
            arrow = "▲" if p in ("golden_cross","hammer","bullish_engulfing","rsi_oversold","macd_bull") else "▼" if p in ("death_cross","shooting_star","bearish_engulfing","rsi_overbought","macd_bear") else "─"
            print(f"  {d}  {arrow} {p:<25}  ${pr:,.2f}")
    else:
        print("  （なし）")

    print("\n" + "─"*65)
    print(f"【{FWD}日後勝率 TOP3（出現10回以上）】")
    print("─"*65)
    top = (stats[stats["出現回数"]>=10]
           .dropna(subset=[f"{FWD}日後勝率%"])
           .nlargest(3, f"{FWD}日後勝率%"))
    for _,row in top.iterrows():
        print(f"  {row['パターン']:<25}  勝率:{row[f'{FWD}日後勝率%']:>5.1f}%  "
              f"平均:{row[f'{FWD}日後平均%']:>+6.3f}%  出現率:{row['出現率%']:.2f}%")

    print("\n" + "═"*65)


if __name__ == "__main__":
    main()
