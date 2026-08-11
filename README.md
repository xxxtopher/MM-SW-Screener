# Stock Screener (Minervini / Weinstein style)

A GitHub-hosted technical stock screener, inspired by Mark Minervini's Trend
Template / VCP concepts and Stan Weinstein's Stage Analysis.

## Status

✅ Full pipeline + dashboard built. Current pieces:

- [x] Repo skeleton
- [x] `src/fetch_universe.py` - sources the ticker universe (S&P 500 + Russell
      3000 via iShares IWV holdings, with a Wikipedia S&P fallback)
- [x] `.github/workflows/refresh_universe.yml` - monthly universe refresh
- [x] `src/fetch_prices.py` - batch daily OHLCV download via yfinance
- [x] `src/criteria.py` - criteria 1-4 (SMA trend template, 52-week range,
      relative strength vs SPX)
- [x] `src/vcp.py` - criteria 5 (weekly range + volume contraction / VCP)
- [x] `src/screen.py` - orchestrates the full pipeline, writes
      `output/screen_results.json`
- [x] `.github/workflows/daily_screen.yml` - runs fetch_prices.py + screen.py
      daily after market close, auto-commits results
- [x] `index.html` - dashboard (static page reading
      `output/screen_results.json`, with per-stock mini candlestick charts)

## Screening criteria

1. **Trend template** - Close > 50-day SMA > ... wait, close above 50, 150,
   and 200-day SMA independently.
2. **Long-term trend rising** - 150-day and 200-day SMA today > value 40
   trading days ago.
3. **52-week range position** - price at least 30% above the 52-week low,
   and no more than 30% below the 52-week high.
4. **Relative strength** - 3-month return beats SPX's 3-month return over
   the same window.
5. **VCP (volatility contraction)** - weekly high-low range monotonically
   decreasing over the last 4 weeks, with weekly volume also monotonically
   decreasing over the same period. Only evaluated on stocks that pass
   criteria 1-4, to keep the pipeline fast.

## Repo structure

```
stock-screener/
├── .github/workflows/
│   ├── refresh_universe.yml   # monthly: re-fetch ticker universe
│   └── daily_screen.yml       # daily: run full screening pipeline
├── data/
│   ├── universe.csv           # cached ticker list (committed to repo)
│   ├── universe_meta.json     # metadata about the last universe fetch
│   └── daily_prices/          # optional local OHLCV cache (gitignored)
├── src/
│   ├── fetch_universe.py
│   ├── fetch_prices.py
│   ├── criteria.py
│   ├── vcp.py
│   └── screen.py
├── output/
│   └── screen_results.json    # latest screening results (dashboard reads this)
├── requirements.txt
└── README.md
```

## Local usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Refresh the ticker universe (writes data/universe.csv)
python src/fetch_universe.py
```
