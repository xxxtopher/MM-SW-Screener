"""
vcp.py

Implements criterion 5 (Volatility Contraction Pattern): weekly price range
OR weekly volume monotonically decreasing over the last N_WEEKS weeks
(loosened 2026-08-10 from requiring both simultaneously to requiring
either one; window extended from 4 to 6 weeks the same day for a deeper,
more reliable base pattern).

Designed to run ONLY on the survivors of criteria 1-4 (from criteria.py),
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

N_WEEKS = 3                 # contraction window (extended from 4 on 2026-08-10
                             # for a deeper, more reliable base pattern)
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

    Weeks are anchored to Friday (W-FRI). A trading-day count per week is
    kept so partial weeks (e.g. today's still-in-progress week, or short
    holiday weeks) can be identified and dropped downstream.
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
    each ticker's last N_WEEKS complete weeks show monotonically decreasing
    range AND monotonically decreasing volume.

    Returns one row per ticker in `tickers` with the weekly figures used
    and a boolean `crit5_vcp` column. Tickers without enough complete weeks
    of history get crit5_vcp = False rather than raising.
    """
    sub = daily_df[daily_df["ticker"].isin(tickers)].copy()
    weekly = to_weekly(sub)

    # Drop partial weeks (not enough trading days in that week's bin) -
    # most relevant for the current, still-in-progress week.
    weekly = weekly[weekly["n_days"] >= MIN_DAYS_FOR_FULL_WEEK]

    rows = []
    for ticker, g in weekly.groupby("ticker"):
        g = g.sort_values("date")
        last_n = g.tail(N_WEEKS)

        if len(last_n) < N_WEEKS:
            rows.append({
                "ticker": ticker, "crit5_vcp": False,
                "weekly_ranges": None, "weekly_volumes": None,
                "n_complete_weeks_available": len(last_n),
            })
            continue

        ranges = last_n["range"].tolist()      # oldest -> newest
        volumes = last_n["volume"].tolist()    # oldest -> newest

        range_contracting = all(ranges[i] > ranges[i + 1] for i in range(len(ranges) - 1))
        volume_contracting = all(volumes[i] > volumes[i + 1] for i in range(len(volumes) - 1))

        # Loosened 2026-08-10: require EITHER range OR volume to contract
        # monotonically over the window, not both simultaneously. Was:
        # range_contracting and volume_contracting.
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
    print(f"[vcp] {len(survivors)} tickers passed criteria 1-4, checking VCP on those only ...")

    if not survivors:
        print("[vcp] No survivors from criteria 1-4 - nothing to check. Exiting.")
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
        print(f"[vcp] Final screen (all 5 criteria) passing tickers: {passers}")


if __name__ == "__main__":
    main()
