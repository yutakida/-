# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## リポジトリ概要

TradingViewと連携するMCPサーバー。yfinanceで過去OHLCVデータを取得し、pandas-taでインジケータ計算・パターン検出・バックテストを行う。ClaudeなどのAIアシスタントからMCPツールとして呼び出す。

## コマンド

```bash
# インストール
pip install -e .

# サーバー起動（Claude Desktop等から呼ばれる想定）
trading-mcp

# 開発時のMCP動作確認
mcp dev src/trading_mcp/server.py
```

## 構成

```
src/trading_mcp/
├── server.py      # MCPツール定義（エントリーポイント）
├── data.py        # yfinanceによるOHLCVデータ取得
├── indicators.py  # pandas-taラッパー（SMA/EMA/RSI/MACD/BBands等）
├── patterns.py    # ローソク足・MAクロス・S&R検出
└── backtest.py    # シグナル生成とベクトル化バックテスト
```

## アーキテクチャ

`server.py` がMCPツールのエントリーポイント。各ツールは `data.py` でDataFrameを取得し、`indicators.py` / `patterns.py` / `backtest.py` に処理を委譲してJSONを返す。

**バックテストの仕組み**: `backtest.run()` はシグナル（+1/0/−1）を受け取り終値リターンで検証する。シグナル生成は `ma_crossover_signals` / `rsi_signals` / `bbands_signals` に分離されている。

**返却形式**: 全ツールはJSON文字列を返す。日付はタイムゾーン除去済み（`tz_localize(None)`）。

## 依存ライブラリ

- `mcp[cli]` — MCPサーバーフレームワーク（FastMCP使用）
- `yfinance` — Yahoo Financeから過去データ取得
- `pandas-ta` — テクニカルインジケータ（TA-Lib不要）
- `pandas` / `numpy` — データ処理

## MCPツール一覧

| ツール | 主要パラメータ |
|--------|---------------|
| `get_historical_data` | symbol, period, interval |
| `calculate_indicator` | symbol, indicator, length（他はMACD/Stoch等の個別パラメータ） |
| `detect_patterns` | symbol, pattern_type（candlestick/ma_crossover） |
| `find_support_resistance` | symbol, window, levels |
| `backtest_strategy` | symbol, strategy（ma_crossover/rsi/bbands） |
| `get_ticker_info` | symbol |

## Claude Desktopへの登録

```json
{
  "mcpServers": {
    "trading": {
      "command": "trading-mcp"
    }
  }
}
```
