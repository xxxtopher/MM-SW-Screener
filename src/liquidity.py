"""
liquidity.py

Implements criterion 6 (liquidity gate for momentum trading): market cap
over MIN_MARKET_CAP and 3-month average daily volume over MIN_AVG_VOLUME.

Designed to run only on the final survivors of criteria 1-5 (the smallest
population in the pipeline), since market cap requires one yfinance network
call per ticker and is the most expensive remaining step.

Can be run standalone-ish via compute_liquidity(), or imported into
screen.py.

Usage (for ad-hoc testing):
    python src/liquidity.py AAPL MSFT GE
"""

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parent.parent

MIN_MARKET_CAP = 1_000_000_000   # $1B
MIN_AVG_VOLUME = 50_000         # shares/day, 3-month average
AVG_VOLUME_LOOKBACK_DAYS = 63    # ~3 trading months, matches RS_LOOKBACK_DAYS in criteria.py

MARKET_CAP_PAUSE_SEC = 0.3       # small pause between per-ticker market cap calls
MARKET_CAP_RETRIES = 3
MARKET_CAP_BACKOFF_SEC = (5, 15, 40)  # increasing pause before each retry, per ticker


# ---------------------------------------------------------------------------
# 3-month average volume (cheap - from data already on disk)
# ---------------------------------------------------------------------------

def compute_avg_volume_3m(daily_df: pd.DataFrame, tickers: list[str]) -> dict[str, float]:
    sub = daily_df[daily_df["ticker"].isin(tickers)].sort_values(["ticker", "date"])
    out = {}
    for ticker, g in sub.groupby("ticker"):
        recent = g.tail(AVG_VOLUME_LOOKBACK_DAYS)
        out[ticker] = float(recent["volume"].mean()) if not recent.empty else None
    return out


# ---------------------------------------------------------------------------
# Market cap (one yfinance call per ticker - keep the input list small)
# ---------------------------------------------------------------------------

def _fetch_one_market_cap(ticker: str) -> float | None:
    """
    Fetches a single ticker's market cap via yfinance's lightweight
    `fast_info`, retrying with backoff on failure (e.g. transient rate
    limiting - common on shared-IP environments like GitHub Actions
    runners). Returns None if every attempt fails, rather than raising.
    """
    last_error: Exception | None = None
    for attempt in range(MARKET_CAP_RETRIES):
        try:
            fi = yf.Ticker(ticker).fast_info
            cap = fi.get("market_cap") or fi.get("marketCap")
            return float(cap) if cap else None
        except Exception as exc:  # noqa: BLE001 - retry, then give up gracefully
            last_error = exc
            if attempt < MARKET_CAP_RETRIES - 1:
                wait = MARKET_CAP_BACKOFF_SEC[attempt]
                print(
                    f"[liquidity] Market cap fetch for {ticker} failed "
                    f"(attempt {attempt + 1}: {exc}); retrying in {wait}s ...",
                    file=sys.stderr,
                )
                time.sleep(wait)

    print(f"[liquidity] Giving up on {ticker} after {MARKET_CAP_RETRIES} attempts: {last_error}", file=sys.stderr)
    return None


def fetch_market_caps(tickers: list[str]) -> dict[str, float | None]:
    """
    Fetches market cap for each ticker (with per-ticker retry on failure).
    Returns None for any ticker where market cap couldn't be determined
    after all retries, rather than raising.
    """
    caps: dict[str, float | None] = {}
    for t in tickers:
        caps[t] = _fetch_one_market_cap(t)
        time.sleep(MARKET_CAP_PAUSE_SEC)
    return caps


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------

def compute_liquidity(daily_df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Returns one row per ticker with market_cap, avg_volume_3m, and a
    crit6_liquidity boolean (True only if both thresholds are met).
    Tickers with unknown market cap (fetch failure) are treated as failing
    this criterion rather than silently passing.
    """
    if not tickers:
        return pd.DataFrame(columns=["ticker", "market_cap", "avg_volume_3m", "crit6_liquidity"])

    avg_vol = compute_avg_volume_3m(daily_df, tickers)
    caps = fetch_market_caps(tickers)

    rows = []
    for t in tickers:
        mc = caps.get(t)
        av = avg_vol.get(t)
        passes = (mc is not None and mc >= MIN_MARKET_CAP) and \
                 (av is not None and av >= MIN_AVG_VOLUME)
        rows.append({
            "ticker": t,
            "market_cap": mc,
            "avg_volume_3m": av,
            "crit6_liquidity": bool(passes),
        })

    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


if __name__ == "__main__":
    test_tickers = sys.argv[1:] or ["AAPL", "MSFT"]
    print(f"[liquidity] Fetching market cap for: {test_tickers}")
    print(fetch_market_caps(test_tickers))
