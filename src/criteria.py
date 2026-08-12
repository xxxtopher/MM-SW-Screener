"""
criteria.py

Implements screening criteria 1-4 (everything except VCP) across the full
universe, using data/daily_prices/latest.parquet as input.

Criteria:
  1. Close > 50-day SMA, Close > 150-day SMA, Close > 200-day SMA
  2. 150-day SMA and 200-day SMA today > their value 40 trading days ago
     (i.e. the long-term trend is rising)
  3. Close >= 1.20 x 52-week low, AND Close >= 0.60 x 52-week high
     (at least 20% above the low, no more than 40% below the high)
  4. 3-month return (63 trading days) beats SPX's 3-month return over the
     same window
  5. Close <= 1.28 x 50-day SMA (not more than ~28% extended above the
     50-day SMA - Minervini's anti-chasing guardrail, added 2026-08-10)

Can be run standalone (writes output/criteria_pass.csv) or imported:
    from criteria import compute_criteria
    result_df = compute_criteria(prices_df)

Usage:
    python src/criteria.py
"""

from pathlib import Path

import pandas as pd
import yfinance as yf
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
LATEST_PRICES_PATH = REPO_ROOT / "data" / "daily_prices" / "latest.parquet"
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_CSV = OUTPUT_DIR / "criteria_pass.csv"

SMA_WINDOWS = (50, 150, 200)
TREND_LOOKBACK_DAYS = 30      # "rising" = today's SMA > SMA from 40 trading days ago
RANGE_WINDOW_DAYS = 252       # ~1 trading year, for 52-week high/low
RANGE_MIN_PERIODS = 100       # allow a slightly shorter history before computing 52wk range
RS_LOOKBACK_DAYS = 63         # ~3 trading months, for relative strength vs SPX

ABOVE_LOW_MULT = 1.20         # price >= 1.20 x 52wk low (widened from 1.30 on 2026-08-10)
BELOW_HIGH_MULT = 0.60        # price >= 0.60 x 52wk high, i.e. within 40% of the high (widened from 0.70)

# Anti-chasing guardrail (added 2026-08-10): Minervini's rule of thumb is to
# avoid buying a stock more than ~25-30% above its 50-day SMA, since that
# signals an extended, higher-risk entry even if the trend itself is intact.
MAX_EXTENSION_ABOVE_50SMA = 1.28  # close <= 1.28 x sma50 (i.e. no more than 28% above it)

SPX_TICKER = "^GSPC"
SPX_FETCH_RETRIES = 4
SPX_FETCH_BACKOFF_SEC = (10, 30, 60, 120)  # increasing pause before each retry


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given long-format daily OHLCV (columns: ticker, date, open, high, low,
    close, volume), adds SMA, trend, 52-week range, and 3-month return
    columns, all computed per-ticker via groupby.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    grp = df.groupby("ticker", group_keys=False)

    for w in SMA_WINDOWS:
        df[f"sma{w}"] = grp["close"].transform(lambda s, w=w: s.rolling(w).mean())

    for w in (150, 200):
        df[f"sma{w}_prior"] = grp[f"sma{w}"].transform(lambda s: s.shift(TREND_LOOKBACK_DAYS))

    df["low_52w"] = grp["low"].transform(
        lambda s: s.rolling(RANGE_WINDOW_DAYS, min_periods=RANGE_MIN_PERIODS).min()
    )
    df["high_52w"] = grp["high"].transform(
        lambda s: s.rolling(RANGE_WINDOW_DAYS, min_periods=RANGE_MIN_PERIODS).max()
    )

    df["ret_3m"] = grp["close"].transform(lambda s: s / s.shift(RS_LOOKBACK_DAYS) - 1)

    return df


def fetch_spx_return_3m() -> float:
    """
    Downloads SPX (^GSPC) daily closes over the same lookback horizon and
    returns its most recent 3-month (63 trading day) return as a scalar,
    used as the relative-strength benchmark for every ticker.

    Retries with increasing backoff on failure - Yahoo's rate limiter hits
    shared-IP environments like GitHub Actions runners harder than a home
    connection, so a single extra call at the end of a long run can get
    rate-limited even when the bulk ticker download just succeeded.
    """
    last_error: Exception | None = None

    for attempt in range(SPX_FETCH_RETRIES):
        try:
            spx = yf.download(SPX_TICKER, period="1y", interval="1d", progress=False, auto_adjust=True)
            if spx.empty or len(spx) < RS_LOOKBACK_DAYS + 1:
                raise RuntimeError("Fetched SPX data but it was empty or too short.")

            close = spx["Close"]
            if isinstance(close, pd.DataFrame):  # yfinance sometimes returns a 1-col DataFrame
                close = close.iloc[:, 0]

            latest_close = close.iloc[-1]
            prior_close = close.iloc[-1 - RS_LOOKBACK_DAYS]
            return float(latest_close / prior_close - 1)

        except Exception as exc:  # noqa: BLE001 - retry on anything, surface the last error if all fail
            last_error = exc
            if attempt < SPX_FETCH_RETRIES - 1:
                wait = SPX_FETCH_BACKOFF_SEC[attempt]
                print(f"[criteria] SPX fetch attempt {attempt + 1} failed ({exc}); retrying in {wait}s ...")
                time.sleep(wait)

    raise RuntimeError(
        f"Could not fetch enough SPX history to compute 3-month return "
        f"after {SPX_FETCH_RETRIES} attempts. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Criteria evaluation
# ---------------------------------------------------------------------------

def compute_criteria(df: pd.DataFrame, spx_ret_3m: float | None = None) -> pd.DataFrame:
    """
    Full pipeline: adds indicators, takes the latest row per ticker, flags
    each criterion, and returns one row per ticker with boolean columns
    crit1..crit4 plus pass_all.
    """
    df = add_indicators(df)

    # Evaluate only the most recent trading date available per ticker.
    latest = df.groupby("ticker", as_index=False).tail(1).copy()

    if spx_ret_3m is None:
        spx_ret_3m = fetch_spx_return_3m()

    latest["crit1_above_smas"] = (
        (latest["close"] > latest["sma50"])
        & (latest["close"] > latest["sma150"])
        & (latest["close"] > latest["sma200"])
    )

    latest["crit2_sma_rising"] = (
        (latest["sma150"] > latest["sma150_prior"])
        & (latest["sma200"] > latest["sma200_prior"])
    )

    latest["crit3_52w_range"] = (
        (latest["close"] >= ABOVE_LOW_MULT * latest["low_52w"])
        & (latest["close"] >= BELOW_HIGH_MULT * latest["high_52w"])
    )

    latest["spx_ret_3m"] = spx_ret_3m
    latest["crit4_relative_strength"] = latest["ret_3m"] > spx_ret_3m

    latest["extension_pct"] = latest["close"] / latest["sma50"] - 1
    latest["crit_not_extended"] = latest["close"] <= MAX_EXTENSION_ABOVE_50SMA * latest["sma50"]

    latest["pass_all"] = (
        latest["crit1_above_smas"]
        & latest["crit2_sma_rising"]
        & latest["crit3_52w_range"]
        & latest["crit4_relative_strength"]
        & latest["crit_not_extended"]
    )

    # NaNs (insufficient history) propagate as False in the boolean columns
    # above via pandas' comparison semantics, so short-history tickers are
    # naturally excluded rather than causing errors.
    for col in ["crit1_above_smas", "crit2_sma_rising", "crit3_52w_range",
                "crit4_relative_strength", "crit_not_extended", "pass_all"]:
        latest[col] = latest[col].fillna(False)

    keep_cols = [
        "ticker", "date", "close",
        "sma50", "sma150", "sma200",
        "low_52w", "high_52w", "ret_3m", "spx_ret_3m", "extension_pct",
        "crit1_above_smas", "crit2_sma_rising", "crit3_52w_range",
        "crit4_relative_strength", "crit_not_extended", "pass_all",
    ]
    return latest[keep_cols].sort_values("ticker").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not LATEST_PRICES_PATH.exists():
        raise FileNotFoundError(
            f"{LATEST_PRICES_PATH} not found - run fetch_prices.py first."
        )

    print(f"[criteria] Loading {LATEST_PRICES_PATH} ...")
    prices = pd.read_parquet(LATEST_PRICES_PATH)
    print(f"[criteria] Loaded {len(prices):,} rows across {prices['ticker'].nunique()} tickers")

    print("[criteria] Fetching SPX 3-month return benchmark ...")
    spx_ret_3m = fetch_spx_return_3m()
    print(f"[criteria] SPX 3-month return: {spx_ret_3m:.2%}")

    print("[criteria] Computing indicators and evaluating criteria 1-4 ...")
    result = compute_criteria(prices, spx_ret_3m=spx_ret_3m)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)

    n_pass = int(result["pass_all"].sum())
    print(f"[criteria] {n_pass} of {len(result)} tickers pass all 4 criteria")
    print(f"[criteria] Full results (all tickers, all criteria flags) saved to {OUTPUT_CSV}")

    passers = result[result["pass_all"]]["ticker"].tolist()
    if passers:
        print(f"[criteria] Passing tickers: {passers}")


if __name__ == "__main__":
    main()
