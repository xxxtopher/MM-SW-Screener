"""
vcp.py

Implements criterion 5 (Volatility Contraction Pattern): weekly price range
OR weekly volume monotonically decreasing over the N_WEEKS complete weeks
BEFORE the most recent OFFSET_WEEKS. The current week is excluded from the
contraction check (added 2026-08-10) so a genuine breakout - which
typically shows expanding volume - doesn't get disqualified for not also
contracting.

Designed to run ONLY on the survivors of criteria 1-4c (from criteria.py),
not the full universe - this is the most computationally expensive check,
so keeping it to a small subset is what makes the whole pipeline fast.

Can be run standalone (writes output/vcp_pass.csv) or imported:
    from vcp import compute_vcp
    result_df = compute_vcp(prices_df, tickers)

Usage:
    python src/vcp.py
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
LATEST_PRICES_PATH = REPO_ROOT / "data" / "daily_prices" / "latest.parquet"
CRITERIA_CSV = REPO_ROOT / "output" / "criteria_pass.csv"
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_CSV = OUTPUT_DIR / "vcp_pass.csv"

N_WEEKS = 2                 # contraction window
OFFSET_WEEKS = 1            # exclude the most recent OFFSET_WEEKS from the
                             # contraction check (added 2026-08-10) - lets the
                             # current week show an expanding breakout instead
                             # of being disqualified for not also contracting
MIN_DAYS_FOR_FULL_WEEK = 3  # weeks with fewer trading days than this are
                             # treated as partial (e.g. the current, still-
                             # in-progress week) and dropped


# ---------------------------------------------------------------------------
# Weekly resampling
# ---------------------------------------------------------------------------

def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples daily OHLCV (long format: ticker, date, open, high, low,
    close, volume) into weekly bars, one grouped resample call across the
    whole input (not a per-ticker loop) for speed.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    weekly = (
        df.set_index("date")
          .groupby("ticker")
          .resample("W-FRI")
          .agg(
              high=("high", "max"),
              low=("low", "min"),
              close=("close", "last"),
              volume=("volume", "sum"),
              n_days=("close", "count"),
          )
          .reset_index()
    )

    weekly["range"] = weekly["high"] - weekly["low"]
    return weekly


# ---------------------------------------------------------------------------
# VCP evaluation
# ---------------------------------------------------------------------------

def compute_vcp(daily_df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Filters daily_df to `tickers`, resamples to weekly, and checks whether
    the N_WEEKS complete weeks BEFORE the most recent OFFSET_WEEKS show
    monotonically decreasing range OR monotonically decreasing volume.

    The current week(s) are deliberately excluded from the contraction
    check (added 2026-08-10): a genuine VCP breakout happens AFTER the
    base has tightened, and the breakout itself typically shows EXPANDING
    volume (institutional buying), not contracting. Requiring the most
    recent week to also be contracting was disqualifying stocks in the
    middle of exactly the kind of breakout this criterion should be
    looking for - it also directly fought against short-window relative
    strength gates (a stock can't have a strong last week AND a
    contracting last week at the same time).
    """
    sub = daily_df[daily_df["ticker"].isin(tickers)].copy()
    weekly = to_weekly(sub)

    weekly = weekly[weekly["n_days"] >= MIN_DAYS_FOR_FULL_WEEK]

    rows = []
    for ticker, g in weekly.groupby("ticker"):
        g = g.sort_values("date")
        window = g.tail(N_WEEKS + OFFSET_WEEKS)

        if len(window) < N_WEEKS + OFFSET_WEEKS:
            rows.append({
                "ticker": ticker, "crit5_vcp": False,
                "weekly_ranges": None, "weekly_volumes": None,
                "n_complete_weeks_available": len(window),
            })
            continue

        # Drop the most recent OFFSET_WEEKS - the base-contraction check
        # only applies to the weeks BEFORE the current (potential breakout)
        # week(s).
        last_n = window.iloc[:-OFFSET_WEEKS] if OFFSET_WEEKS > 0 else window

        ranges = last_n["range"].tolist()      # oldest -> newest
        volumes = last_n["volume"].tolist()    # oldest -> newest

        range_contracting = all(ranges[i] > ranges[i + 1] for i in range(len(ranges) - 1))
        volume_contracting = all(volumes[i] > volumes[i + 1] for i in range(len(volumes) - 1))

        rows.append({
            "ticker": ticker,
            "crit5_vcp": bool(range_contracting or volume_contracting),
            "range_contracting": bool(range_contracting),
            "volume_contracting": bool(volume_contracting),
            "weekly_ranges": ranges,
            "weekly_volumes": volumes,
            "n_complete_weeks_available": len(last_n),
        })

    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CRITERIA_CSV.exists():
        raise FileNotFoundError(f"{CRITERIA_CSV} not found - run criteria.py first.")
    if not LATEST_PRICES_PATH.exists():
        raise FileNotFoundError(f"{LATEST_PRICES_PATH} not found - run fetch_prices.py first.")

    criteria_df = pd.read_csv(CRITERIA_CSV)
    survivors = criteria_df[criteria_df["pass_all"]]["ticker"].tolist()
    print(f"[vcp] {len(survivors)} tickers passed criteria 1-4c, checking VCP on those only ...")

    if not survivors:
        print("[vcp] No survivors - nothing to check. Exiting.")
        return

    print(f"[vcp] Loading {LATEST_PRICES_PATH} ...")
    daily = pd.read_parquet(LATEST_PRICES_PATH)

    result = compute_vcp(daily, survivors)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)

    n_pass = int(result["crit5_vcp"].sum())
    print(f"[vcp] {n_pass} of {len(survivors)} survivors also show VCP contraction")
    print(f"[vcp] Full results saved to {OUTPUT_CSV}")

    passers = result[result["crit5_vcp"]]["ticker"].tolist()
    if passers:
        print(f"[vcp] Passing tickers: {passers}")


if __name__ == "__main__":
    main()
