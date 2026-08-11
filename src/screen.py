"""
screen.py

Orchestrates the full screening pipeline:
  1. Load data/daily_prices/latest.parquet
  2. Run criteria 1-4 (criteria.py) across the whole universe
  3. Run criterion 5 / VCP (vcp.py) only on the criteria 1-4 survivors
  4. Combine into a final pass/fail list
  5. For final passers, embed the last ~3 months of daily OHLC (for the
     dashboard's mini price charts) directly into the output JSON so the
     dashboard doesn't need a separate data call per stock
  6. Write output/screen_results.json

This assumes data/daily_prices/latest.parquet is already fresh (i.e.
fetch_prices.py has already been run). This script does not re-fetch
prices itself, to keep the network-heavy step and the compute-heavy step
separately runnable/debuggable.

Usage:
    python src/screen.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from criteria import compute_criteria, fetch_spx_return_3m
from vcp import compute_vcp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
LATEST_PRICES_PATH = REPO_ROOT / "data" / "daily_prices" / "latest.parquet"
OUTPUT_DIR = REPO_ROOT / "output"
RESULTS_JSON = OUTPUT_DIR / "screen_results.json"
CRITERIA_CSV = OUTPUT_DIR / "criteria_pass.csv"
VCP_CSV = OUTPUT_DIR / "vcp_pass.csv"

CHART_LOOKBACK_DAYS = 65   # ~3 trading months, for the dashboard mini charts


# ---------------------------------------------------------------------------
# Chart data
# ---------------------------------------------------------------------------

def build_chart_data(daily_df: pd.DataFrame, tickers: list[str]) -> dict[str, list[dict]]:
    """
    For each ticker, returns its last CHART_LOOKBACK_DAYS daily bars as a
    list of {date, open, high, low, close, volume} dicts, ready to embed
    directly into the results JSON for client-side chart rendering.
    """
    sub = daily_df[daily_df["ticker"].isin(tickers)].sort_values(["ticker", "date"])

    charts: dict[str, list[dict]] = {}
    for ticker, g in sub.groupby("ticker"):
        recent = g.tail(CHART_LOOKBACK_DAYS)
        charts[ticker] = [
            {
                "date": row.date.strftime("%Y-%m-%d"),
                "open": round(float(row.open), 4),
                "high": round(float(row.high), 4),
                "low": round(float(row.low), 4),
                "close": round(float(row.close), 4),
                "volume": int(row.volume),
            }
            for row in recent.itertuples(index=False)
        ]
    return charts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not LATEST_PRICES_PATH.exists():
        raise FileNotFoundError(
            f"{LATEST_PRICES_PATH} not found - run fetch_prices.py first."
        )

    print(f"[screen] Loading {LATEST_PRICES_PATH} ...")
    daily = pd.read_parquet(LATEST_PRICES_PATH)
    print(f"[screen] Loaded {len(daily):,} rows across {daily['ticker'].nunique()} tickers")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[screen] Fetching SPX 3-month return benchmark ...")
    spx_ret_3m = fetch_spx_return_3m()
    print(f"[screen] SPX 3-month return: {spx_ret_3m:.2%}")

    print("[screen] Running criteria 1-4 across the full universe ...")
    criteria_result = compute_criteria(daily, spx_ret_3m=spx_ret_3m)
    criteria_result.to_csv(CRITERIA_CSV, index=False)

    survivors = criteria_result[criteria_result["pass_all"]]["ticker"].tolist()
    print(f"[screen] {len(survivors)} of {len(criteria_result)} pass criteria 1-4")

    if not survivors:
        print("[screen] No survivors from criteria 1-4 - writing empty result set.")
        vcp_result = pd.DataFrame(columns=["ticker", "crit5_vcp"])
        final_tickers: list[str] = []
    else:
        print(f"[screen] Running VCP check (criterion 5) on {len(survivors)} survivors ...")
        vcp_result = compute_vcp(daily, survivors)
        vcp_result.to_csv(VCP_CSV, index=False)
        final_tickers = vcp_result[vcp_result["crit5_vcp"]]["ticker"].tolist()

    print(f"[screen] {len(final_tickers)} tickers pass all 5 criteria: {final_tickers}")

    print(f"[screen] Building {CHART_LOOKBACK_DAYS}-day chart data for the dashboard ...")
    chart_data = build_chart_data(daily, final_tickers)

    # Merge summary stats (from criteria_result) for each final passer,
    # so the dashboard has key numbers to display alongside each chart
    # without needing to re-derive them client-side.
    summary_cols = [
        "ticker", "close", "sma50", "sma150", "sma200",
        "low_52w", "high_52w", "ret_3m", "spx_ret_3m",
    ]
    summary_lookup = (
        criteria_result[criteria_result["ticker"].isin(final_tickers)][summary_cols]
        .set_index("ticker")
        .to_dict(orient="index")
    )

    stocks_out = []
    for ticker in final_tickers:
        stats = summary_lookup.get(ticker, {})
        stocks_out.append({
            "ticker": ticker,
            "close": stats.get("close"),
            "sma50": stats.get("sma50"),
            "sma150": stats.get("sma150"),
            "sma200": stats.get("sma200"),
            "low_52w": stats.get("low_52w"),
            "high_52w": stats.get("high_52w"),
            "ret_3m": stats.get("ret_3m"),
            "spx_ret_3m": stats.get("spx_ret_3m"),
            "chart": chart_data.get(ticker, []),
        })

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe_size": int(daily["ticker"].nunique()),
        "criteria_1_4_pass_count": len(survivors),
        "final_pass_count": len(final_tickers),
        "spx_ret_3m": spx_ret_3m,
        "stocks": stocks_out,
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"[screen] Wrote final results (with chart data) to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
