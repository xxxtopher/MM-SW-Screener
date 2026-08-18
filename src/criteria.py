"""
criteria.py

Implements screening criteria 1-4c (everything except VCP and liquidity)
across the full universe, using data/daily_prices/latest.parquet as input.

Criteria:
  1. Close > 50-day SMA, Close > 150-day SMA, Close > 200-day SMA
  2. 150-day SMA today > its value 30 trading days ago; 200-day SMA today
     > its value 150 trading days ago
  3. Close >= 1.20 x 52-week low, AND Close >= 0.60 x 52-week high
     (at least 20% above the low, no more than 40% below the high)
  4. 1-month return beats SPX's 1-month return by at least RS_MARGIN_1M
  4b. 2-week return beats SPX's 2-week return by at least RS_MARGIN_2W
  4c. 3-month return beats SPX's 3-month return by at least RS_MARGIN_3M
      (added back 2026-08-10, alongside explicit margins on 4/4b, to
       tighten relative strength into a genuine multi-timeframe gate
       rather than "beats SPX by any amount")
  5. Close <= 1.28 x 50-day SMA (anti-chasing guardrail)
  6. |Close - 20-day SMA| / 20-day SMA <= 15% (tightness/consolidation)

Also computes `alpha_score` for every ticker (regardless of pass/fail) - a
weighted blend of excess return over SPX across the three RS horizons.
screen.py uses this to rank final survivors and show only the top N,
rather than every stock that technically clears the hard gates.

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

SMA_WINDOWS = (20, 50, 150, 200)
TREND_LOOKBACK_DAYS_150 = 30   # "rising" = today's 150-day SMA > its value 30 trading days ago
TREND_LOOKBACK_DAYS_200 = 150  # "rising" = today's 200-day SMA > its value 150 trading days ago

RANGE_WINDOW_DAYS = 252       # ~1 trading year, for 52-week high/low
RANGE_MIN_PERIODS = 100       # allow a slightly shorter history before computing 52wk range

RS_LOOKBACK_DAYS = 21          # ~1 trading month
RS_LOOKBACK_DAYS_2W = 10       # ~2 trading weeks
RS_LOOKBACK_DAYS_3M = 63       # ~3 trading months (re-added 2026-08-10)

# Margin-based RS gates (added 2026-08-10): previously a stock passed by
# beating SPX by ANY amount, even 0.01%. Requiring an explicit minimum
# margin on each horizon makes this a genuine outperformance filter rather
# than a coin-flip around the benchmark.
RS_MARGIN_1M = 0.01   # must beat SPX's 1-month return by >= 1 percentage point (loosened from 3pp on 2026-08-10)
RS_MARGIN_2W = 0.005  # must beat SPX's 2-week return by >= 0.5 percentage points (loosened from 2pp on 2026-08-10)
RS_MARGIN_3M = 0.02   # must beat SPX's 3-month return by >= 2 percentage points (loosened from 5pp on 2026-08-10)

# Alpha Score weights (added 2026-08-10): used to rank survivors, not to
# gate them. Weights sum to 1.0. Weighted toward 1-month and 3-month
# (recency + stability) with less weight on the noisier 2-week window.
ALPHA_WEIGHT_1M = 0.4
ALPHA_WEIGHT_2W = 0.2
ALPHA_WEIGHT_3M = 0.4

ABOVE_LOW_MULT = 1.20         # price >= 1.20 x 52wk low
BELOW_HIGH_MULT = 0.60        # price >= 0.60 x 52wk high, i.e. within 40% of the high

# Anti-chasing guardrail: Minervini's rule of thumb is to avoid buying a
# stock more than ~25-30% above its 50-day SMA.
MAX_EXTENSION_ABOVE_50SMA = 1.28  # close <= 1.28 x sma50

# Tightness filter: require price to be consolidating close to its 20-day
# SMA, not running away from it.
MAX_DIST_FROM_SMA20 = 0.15    # |close - sma20| / sma20 <= 15%

SPX_TICKER = "^GSPC"
SPX_FETCH_RETRIES = 4
SPX_FETCH_BACKOFF_SEC = (10, 30, 60, 120)  # increasing pause before each retry


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given long-format daily OHLCV (columns: ticker, date, open, high, low,
    close, volume), adds SMA, trend, 52-week range, and multi-horizon
    return columns, all computed per-ticker via groupby.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    grp = df.groupby("ticker", group_keys=False)

    for w in SMA_WINDOWS:
        df[f"sma{w}"] = grp["close"].transform(lambda s, w=w: s.rolling(w).mean())

    df["sma150_prior"] = grp["sma150"].transform(lambda s: s.shift(TREND_LOOKBACK_DAYS_150))
    df["sma200_prior"] = grp["sma200"].transform(lambda s: s.shift(TREND_LOOKBACK_DAYS_200))

    df["low_52w"] = grp["low"].transform(
        lambda s: s.rolling(RANGE_WINDOW_DAYS, min_periods=RANGE_MIN_PERIODS).min()
    )
    df["high_52w"] = grp["high"].transform(
        lambda s: s.rolling(RANGE_WINDOW_DAYS, min_periods=RANGE_MIN_PERIODS).max()
    )

    df["ret_1m"] = grp["close"].transform(lambda s: s / s.shift(RS_LOOKBACK_DAYS) - 1)
    df["ret_2w"] = grp["close"].transform(lambda s: s / s.shift(RS_LOOKBACK_DAYS_2W) - 1)
    df["ret_3m"] = grp["close"].transform(lambda s: s / s.shift(RS_LOOKBACK_DAYS_3M) - 1)

    return df


def fetch_spx_returns() -> dict[str, float]:
    """
    Downloads SPX (^GSPC) daily closes and returns its most recent 1-month,
    2-week, and 3-month returns as a dict, computed from a single download
    (avoids separate network calls per horizon).

    Retries with increasing backoff on failure - Yahoo's rate limiter hits
    shared-IP environments like GitHub Actions runners harder than a home
    connection.
    """
    last_error: Exception | None = None
    min_len = max(RS_LOOKBACK_DAYS, RS_LOOKBACK_DAYS_2W, RS_LOOKBACK_DAYS_3M) + 1

    for attempt in range(SPX_FETCH_RETRIES):
        try:
            spx = yf.download(SPX_TICKER, period="1y", interval="1d", progress=False, auto_adjust=True)
            if spx.empty or len(spx) < min_len:
                raise RuntimeError("Fetched SPX data but it was empty or too short.")

            close = spx["Close"]
            if isinstance(close, pd.DataFrame):  # yfinance sometimes returns a 1-col DataFrame
                close = close.iloc[:, 0]

            latest_close = close.iloc[-1]
            ret_1m = float(latest_close / close.iloc[-1 - RS_LOOKBACK_DAYS] - 1)
            ret_2w = float(latest_close / close.iloc[-1 - RS_LOOKBACK_DAYS_2W] - 1)
            ret_3m = float(latest_close / close.iloc[-1 - RS_LOOKBACK_DAYS_3M] - 1)
            return {"ret_1m": ret_1m, "ret_2w": ret_2w, "ret_3m": ret_3m}

        except Exception as exc:  # noqa: BLE001 - retry on anything, surface the last error if all fail
            last_error = exc
            if attempt < SPX_FETCH_RETRIES - 1:
                wait = SPX_FETCH_BACKOFF_SEC[attempt]
                print(f"[criteria] SPX fetch attempt {attempt + 1} failed ({exc}); retrying in {wait}s ...")
                time.sleep(wait)

    raise RuntimeError(
        f"Could not fetch SPX returns after {SPX_FETCH_RETRIES} attempts. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Criteria evaluation
# ---------------------------------------------------------------------------

def compute_criteria(
    df: pd.DataFrame,
    spx_ret_1m: float | None = None,
    spx_ret_2w: float | None = None,
    spx_ret_3m: float | None = None,
) -> pd.DataFrame:
    """
    Full pipeline: adds indicators, takes the latest row per ticker, flags
    each criterion, computes alpha_score, and returns one row per ticker.
    """
    df = add_indicators(df)

    # Evaluate only the most recent trading date available per ticker.
    latest = df.groupby("ticker", as_index=False).tail(1).copy()

    if spx_ret_1m is None or spx_ret_2w is None or spx_ret_3m is None:
        spx_returns = fetch_spx_returns()
        spx_ret_1m = spx_returns["ret_1m"] if spx_ret_1m is None else spx_ret_1m
        spx_ret_2w = spx_returns["ret_2w"] if spx_ret_2w is None else spx_ret_2w
        spx_ret_3m = spx_returns["ret_3m"] if spx_ret_3m is None else spx_ret_3m

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

    latest["spx_ret_1m"] = spx_ret_1m
    latest["spx_ret_2w"] = spx_ret_2w
    latest["spx_ret_3m"] = spx_ret_3m

    latest["excess_1m"] = latest["ret_1m"] - spx_ret_1m
    latest["excess_2w"] = latest["ret_2w"] - spx_ret_2w
    latest["excess_3m"] = latest["ret_3m"] - spx_ret_3m

    latest["crit4_relative_strength"] = latest["excess_1m"] >= RS_MARGIN_1M
    latest["crit4b_outperform_2w"] = latest["excess_2w"] >= RS_MARGIN_2W
    latest["crit4c_relative_strength_3m"] = latest["excess_3m"] >= RS_MARGIN_3M

    latest["extension_pct"] = latest["close"] / latest["sma50"] - 1
    latest["crit_not_extended"] = latest["close"] <= MAX_EXTENSION_ABOVE_50SMA * latest["sma50"]

    latest["dist_from_sma20_pct"] = (latest["close"] - latest["sma20"]).abs() / latest["sma20"]
    latest["crit_near_sma20"] = latest["dist_from_sma20_pct"] <= MAX_DIST_FROM_SMA20

    latest["alpha_score"] = (
        ALPHA_WEIGHT_1M * latest["excess_1m"]
        + ALPHA_WEIGHT_2W * latest["excess_2w"]
        + ALPHA_WEIGHT_3M * latest["excess_3m"]
    )

    latest["pass_all"] = (
        latest["crit1_above_smas"]
        & latest["crit2_sma_rising"]
        & latest["crit3_52w_range"]
        & latest["crit4_relative_strength"]
        & latest["crit4b_outperform_2w"]
        & latest["crit4c_relative_strength_3m"]
        & latest["crit_not_extended"]
        & latest["crit_near_sma20"]
    )

    # NaNs (insufficient history) propagate as False in the boolean columns
    # above via pandas' comparison semantics, so short-history tickers are
    # naturally excluded rather than causing errors.
    bool_cols = [
        "crit1_above_smas", "crit2_sma_rising", "crit3_52w_range",
        "crit4_relative_strength", "crit4b_outperform_2w", "crit4c_relative_strength_3m",
        "crit_not_extended", "crit_near_sma20", "pass_all",
    ]
    for col in bool_cols:
        latest[col] = latest[col].fillna(False)

    keep_cols = [
        "ticker", "date", "close",
        "sma20", "sma50", "sma150", "sma200",
        "low_52w", "high_52w",
        "ret_1m", "spx_ret_1m", "excess_1m",
        "ret_2w", "spx_ret_2w", "excess_2w",
        "ret_3m", "spx_ret_3m", "excess_3m",
        "extension_pct", "dist_from_sma20_pct", "alpha_score",
        "crit1_above_smas", "crit2_sma_rising", "crit3_52w_range",
        "crit4_relative_strength", "crit4b_outperform_2w", "crit4c_relative_strength_3m",
        "crit_not_extended", "crit_near_sma20", "pass_all",
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

    print("[criteria] Fetching SPX 1-month, 2-week, and 3-month return benchmarks ...")
    spx_returns = fetch_spx_returns()
    print(f"[criteria] SPX 1-month return: {spx_returns['ret_1m']:.2%}")
    print(f"[criteria] SPX 2-week return: {spx_returns['ret_2w']:.2%}")
    print(f"[criteria] SPX 3-month return: {spx_returns['ret_3m']:.2%}")

    print("[criteria] Computing indicators and evaluating criteria ...")
    result = compute_criteria(
        prices,
        spx_ret_1m=spx_returns["ret_1m"],
        spx_ret_2w=spx_returns["ret_2w"],
        spx_ret_3m=spx_returns["ret_3m"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)

    n_pass = int(result["pass_all"].sum())
    print(f"[criteria] {n_pass} of {len(result)} tickers pass all criteria")
    print(f"[criteria] Full results saved to {OUTPUT_CSV}")

    passers = result[result["pass_all"]].sort_values("alpha_score", ascending=False)["ticker"].tolist()
    if passers:
        print(f"[criteria] Passing tickers (ranked by alpha_score): {passers}")


if __name__ == "__main__":
    main()
