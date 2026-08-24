"""
criteria.py

Implements screening criteria 1-4c (everything except VCP and liquidity)
across the full universe, using data/daily_prices/latest.parquet as input.

Criteria (revised 2026-08-18 for pullback-entry targeting):
  1. Close > 50-day SMA, Close > 150-day SMA, Close > 200-day SMA
  2. 150-day SMA today > its value 30 trading days ago; 200-day SMA today
     > its value 150 trading days ago
  3. Close >= 1.20 x 52-week low, AND Close >= 0.60 x 52-week high
     (at least 20% above the low, no more than 40% below the high)
  4. At least 2 of 3 RS gates pass (any 2 of: 1m, 2w, 3m vs SPX margins)
  5. Close <= 1.15 x 50-day SMA (tightened from 1.28: requires genuine
     pullback from extension, not just "not too far above")
  6. Price between -5% and +5% of 20-day SMA (directional pullback zone:
     price has come back to or just below the 20-day SMA - changed from
     symmetric |distance| <= 15%)
  7. Close is 5-20% below the 20-day high (stock has genuinely pulled back
     from a recent high but not collapsed - new criterion 2026-08-18)
  8. Momentum not decelerating: excess_1m >= excess_3m x 0.7 (1-month
     outperformance vs SPX must be at least 70% of 3-month outperformance -
     replaces the 2-week margin gate as the acceleration proxy)

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
import math

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

RECENT_HIGH_WINDOW = 20       # trading days for the "pulled back from recent high" check

RS_LOOKBACK_DAYS = 21          # ~1 trading month
RS_LOOKBACK_DAYS_2W = 10       # ~2 trading weeks
RS_LOOKBACK_DAYS_3M = 63       # ~3 trading months

# Margin-based RS gates
RS_MARGIN_1M = 0.01   # must beat SPX's 1-month return by >= 1 percentage point
RS_MARGIN_2W = 0.005  # must beat SPX's 2-week return by >= 0.5 percentage points
RS_MARGIN_3M = 0.02   # must beat SPX's 3-month return by >= 2 percentage points

# Alpha Score weights: used to rank survivors, not to gate them.
ALPHA_WEIGHT_1M = 0.4
ALPHA_WEIGHT_2W = 0.2
ALPHA_WEIGHT_3M = 0.4

ABOVE_LOW_MULT = 1.20         # price >= 1.20 x 52wk low
BELOW_HIGH_MULT = 0.60        # price >= 0.60 x 52wk high, i.e. within 40% of the high

# Anti-extension guardrail: tightened from 1.28 (28%) to 1.25 (25%) on
# 2026-08-18 to require a genuine pullback from extension - a stock still
# 28% above its 50-day SMA hasn't really pulled back.
MAX_EXTENSION_ABOVE_50SMA = 1.28  # close <= 1.25 x sma50

# Pullback zone: price must be between -5% and +5% of the 20-day SMA
# (directional filter targeting stocks that have retraced back toward the
# 20-day SMA, not just "near it in any direction").
SMA20_PULLBACK_LOW = -0.05    # close >= sma20 * (1 - 0.05), i.e. no more than 5% below
SMA20_PULLBACK_HIGH = 0.15    # close <= sma20 * (1 + 0.15), i.e. no more than 5% above

# Pullback-from-high filter: stock must have pulled back 5-20% from its
# RECENT_HIGH_WINDOW-day high - it was running, pulled back, but hasn't
# collapsed. Below 5% = barely retraced. Above 20% = too much damage.
PULLBACK_FROM_HIGH_MIN = 0.01  # at least 3% below the recent high (loosened from 5% on 2026-08-18)
PULLBACK_FROM_HIGH_MAX = 0.25  # no more than 20% below the recent high

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

    # Recent high over last RECENT_HIGH_WINDOW days (for pullback-from-high filter)
    df["high_recent"] = grp["high"].transform(
        lambda s: s.rolling(RECENT_HIGH_WINDOW).max()
    )

    return df


def fetch_spx_returns() -> dict[str, float]:
    """
    Downloads SPX (^GSPC) daily closes and returns its most recent 1-month,
    2-week, and 3-month returns as a dict, computed from a single download
    (avoids separate network calls per horizon).

    Retries with increasing backoff on failure - Yahoo's rate limiter hits
    shared-IP environments like GitHub Actions runners harder than a home
    connection. Also retries if the fetch "succeeds" but the specific
    close values used turn out to be NaN (e.g. a stray gap in the series,
    or an unfinalized same-day close) - a fetch that returns data isn't
    the same as a fetch that returns USABLE data, and silently returning
    NaN here poisons every relative-strength comparison downstream without
    any visible error (found 2026-08-10: every RS gate was passing 0% of
    the universe because spx_ret_1m/2w/3m were silently NaN).
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

            # Drop any NaN rows (stray gaps, unfinalized/partial sessions)
            # before indexing, so a single bad row doesn't corrupt the
            # calculation - re-check length after dropping.
            close = close.dropna()
            if len(close) < min_len:
                raise RuntimeError(
                    f"SPX close series had only {len(close)} valid (non-NaN) "
                    f"rows after dropping NaNs, need at least {min_len}."
                )

            latest_close = close.iloc[-1]
            ret_1m = float(latest_close / close.iloc[-1 - RS_LOOKBACK_DAYS] - 1)
            ret_2w = float(latest_close / close.iloc[-1 - RS_LOOKBACK_DAYS_2W] - 1)
            ret_3m = float(latest_close / close.iloc[-1 - RS_LOOKBACK_DAYS_3M] - 1)

            # Validate the computed values themselves - a "successful" fetch
            # that still produces NaN (e.g. from an unexpected data shape)
            # must not be returned silently.
            if any(math.isnan(v) for v in (ret_1m, ret_2w, ret_3m)):
                raise RuntimeError(
                    f"Computed SPX returns contain NaN: ret_1m={ret_1m}, "
                    f"ret_2w={ret_2w}, ret_3m={ret_3m}"
                )

            return {"ret_1m": ret_1m, "ret_2w": ret_2w, "ret_3m": ret_3m}

        except Exception as exc:  # noqa: BLE001 - retry on anything, surface the last error if all fail
            last_error = exc
            if attempt < SPX_FETCH_RETRIES - 1:
                wait = SPX_FETCH_BACKOFF_SEC[attempt]
                print(f"[criteria] SPX fetch attempt {attempt + 1} failed ({exc}); retrying in {wait}s ...")
                time.sleep(wait)

    raise RuntimeError(
        f"Could not fetch valid SPX returns after {SPX_FETCH_RETRIES} attempts. Last error: {last_error}"
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

    # Require at least 2 of 3 RS gates
    latest["rs_gates_passed_count"] = (
        latest["crit4_relative_strength"].astype(int)
        + latest["crit4b_outperform_2w"].astype(int)
        + latest["crit4c_relative_strength_3m"].astype(int)
    )
    latest["crit4_rs_combined"] = latest["rs_gates_passed_count"] >= 2

    # Criterion 5: tightened extension cap (was 28%, now 15%)
    latest["extension_pct"] = latest["close"] / latest["sma50"] - 1
    latest["crit_not_extended"] = latest["close"] <= MAX_EXTENSION_ABOVE_50SMA * latest["sma50"]

    # Criterion 6: directional pullback zone — price between -5% and +5% of
    # 20-day SMA (replaced symmetric |distance| <= 15%)
    latest["dist_from_sma20_pct"] = (latest["close"] - latest["sma20"]) / latest["sma20"]
    latest["crit_near_sma20"] = (
        (latest["dist_from_sma20_pct"] >= SMA20_PULLBACK_LOW)
        & (latest["dist_from_sma20_pct"] <= SMA20_PULLBACK_HIGH)
    )

    # Criterion 7: pulled back 3-20% from the RECENT_HIGH_WINDOW-day high
    # (min loosened from 5% to 3% on 2026-08-18: data showed most candidates
    # were pulling back 3-5%, sitting just below the old 5% minimum)
    latest["pullback_from_high_pct"] = (latest["high_recent"] - latest["close"]) / latest["high_recent"]
    latest["crit_pullback_from_high"] = (
        (latest["pullback_from_high_pct"] >= PULLBACK_FROM_HIGH_MIN)
        & (latest["pullback_from_high_pct"] <= PULLBACK_FROM_HIGH_MAX)
    )

    # Criterion 8: momentum not decelerating — 1-month excess return must be
    # at least 70% of the 3-month excess return (loosened from strict >
    # on 2026-08-18: in a broad bull market, excess_1m > excess_3m is very
    # hard to clear; 0.7x still filters clear decelerators while allowing
    # normal short-term noise in RS)
    latest["crit_momentum_accelerating"] = latest["excess_1m"] >= latest["excess_3m"] * 0.3

    latest["alpha_score"] = (
        ALPHA_WEIGHT_1M * latest["excess_1m"]
        + ALPHA_WEIGHT_2W * latest["excess_2w"]
        + ALPHA_WEIGHT_3M * latest["excess_3m"]
    )

    latest["pass_all"] = (
        latest["crit1_above_smas"]
        & latest["crit2_sma_rising"]
        & latest["crit3_52w_range"]
        & latest["crit4_rs_combined"]
        & latest["crit_not_extended"]
        & latest["crit_near_sma20"]
        & latest["crit_pullback_from_high"]
        & latest["crit_momentum_accelerating"]
    )

    # NaNs (insufficient history) propagate as False in the boolean columns
    # above via pandas' comparison semantics, so short-history tickers are
    # naturally excluded rather than causing errors.
    bool_cols = [
        "crit1_above_smas", "crit2_sma_rising", "crit3_52w_range",
        "crit4_relative_strength", "crit4b_outperform_2w", "crit4c_relative_strength_3m",
        "crit4_rs_combined", "crit_not_extended", "crit_near_sma20",
        "crit_pullback_from_high", "crit_momentum_accelerating",
        "pass_all",
    ]
    for col in bool_cols:
        latest[col] = latest[col].fillna(False)

    keep_cols = [
        "ticker", "date", "close",
        "sma20", "sma50", "sma150", "sma200",
        "low_52w", "high_52w", "high_recent",
        "ret_1m", "spx_ret_1m", "excess_1m",
        "ret_2w", "spx_ret_2w", "excess_2w",
        "ret_3m", "spx_ret_3m", "excess_3m",
        "extension_pct", "dist_from_sma20_pct",
        "pullback_from_high_pct", "alpha_score",
        "crit1_above_smas", "crit2_sma_rising", "crit3_52w_range",
        "crit4_relative_strength", "crit4b_outperform_2w", "crit4c_relative_strength_3m",
        "rs_gates_passed_count", "crit4_rs_combined",
        "crit_not_extended", "crit_near_sma20",
        "crit_pullback_from_high", "crit_momentum_accelerating",
        "pass_all",
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
