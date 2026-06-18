# trading-mcp

過去データのパターン検証・インジケータ計算を行うMCPサーバー。Claude等のAIアシスタントからトレーディング分析ツールを呼び出せます。

## セットアップ

```bash
pip install -e .
```

## Claude Desktopへの登録

`~/Library/Application Support/Claude/claude_desktop_config.json` に追加：

```json
{
  "mcpServers": {
    "trading": {
      "command": "trading-mcp"
    }
  }
}
```

## 提供ツール

| ツール | 説明 |
|--------|------|
| `get_historical_data` | 過去OHLCVデータ取得 |
| `calculate_indicator` | テクニカルインジケータ計算 (SMA/EMA/RSI/MACD/BBands/ATR/Stoch/ADX/VWAP) |
| `detect_patterns` | ローソク足・MAクロスパターン検出 |
| `find_support_resistance` | サポート・レジスタンスライン検出 |
| `backtest_strategy` | 戦略バックテスト (MA/RSI/BBands) |
| `get_ticker_info` | 銘柄基本情報取得 |

## 対応銘柄

yfinanceが対応する全銘柄。例：
- 米国株: `AAPL`, `TSLA`, `SPY`
- 日本株: `7203.T`（トヨタ）, `^N225`（日経平均）
- 暗号資産: `BTC-USD`, `ETH-USD`
- FX: `JPY=X`（ドル円）
