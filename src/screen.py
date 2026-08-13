"""
screen.py

Orchestrates the full screening pipeline:
  1. Load data/daily_prices/latest.parquet
  2. Run criteria 1-4 (criteria.py) across the whole universe
  3. Run criterion 5 / VCP (vcp.py) only on the criteria 1-4 survivors
  4. Run criterion 6 / liquidity gate (liquidity.py) only on the VCP
     survivors - market cap and 3-month average volume, for momentum
     trading practicality
  5. Combine into a final pass/fail list
  6. For final passers, embed the last ~3 months of daily OHLC (for the
     dashboard's mini price charts) directly into the output JSON so the
     dashboard doesn't need a separate data call per stock
  7. Write output/screen_results.json

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

from criteria import compute_criteria, fetch_spx_returns
from vcp import compute_vcp
from liquidity import compute_liquidity, MIN_MARKET_CAP, MIN_AVG_VOLUME

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
LATEST_PRICES_PATH = REPO_ROOT / "data" / "daily_prices" / "latest.parquet"
OUTPUT_DIR = REPO_ROOT / "output"
RESULTS_JSON = OUTPUT_DIR / "screen_results.json"
CRITERIA_CSV = OUTPUT_DIR / "criteria_pass.csv"
VCP_CSV = OUTPUT_DIR / "vcp_pass.csv"
LIQUIDITY_CSV = OUTPUT_DIR / "liquidity_pass.csv"

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

    print("[screen] Fetching SPX 1-month and 1-week return benchmarks ...")
    spx_returns = fetch_spx_returns()
    spx_ret_1m = spx_returns["ret_1m"]
    spx_ret_1w = spx_returns["ret_1w"]
    print(f"[screen] SPX 1-month return: {spx_ret_1m:.2%}")
    print(f"[screen] SPX 1-week return: {spx_ret_1w:.2%}")

    print("[screen] Running criteria 1-4 across the full universe ...")
    criteria_result = compute_criteria(daily, spx_ret_1m=spx_ret_1m, spx_ret_1w=spx_ret_1w)
    criteria_result.to_csv(CRITERIA_CSV, index=False)

    survivors = criteria_result[criteria_result["pass_all"]]["ticker"].tolist()
    print(f"[screen] {len(survivors)} of {len(criteria_result)} pass criteria 1-4")

    if not survivors:
        print("[screen] No survivors from criteria 1-4 - writing empty result set.")
        vcp_result = pd.DataFrame(columns=["ticker", "crit5_vcp"])
        vcp_passers: list[str] = []
    else:
        print(f"[screen] Running VCP check (criterion 5) on {len(survivors)} survivors ...")
        vcp_result = compute_vcp(daily, survivors)
        vcp_result.to_csv(VCP_CSV, index=False)
        vcp_passers = vcp_result[vcp_result["crit5_vcp"]]["ticker"].tolist()

    print(f"[screen] {len(vcp_passers)} tickers pass criteria 1-5 (before liquidity gate)")

    if not vcp_passers:
        print("[screen] No VCP survivors - skipping liquidity check.")
        liquidity_result = pd.DataFrame(columns=["ticker", "market_cap", "avg_volume_3m", "crit6_liquidity"])
        final_tickers: list[str] = []
    else:
        print(
            f"[screen] Checking liquidity (market cap >= ${MIN_MARKET_CAP:,.0f}, "
            f"3-month avg volume >= {MIN_AVG_VOLUME:,}) on {len(vcp_passers)} survivors ..."
        )
        liquidity_result = compute_liquidity(daily, vcp_passers)
        liquidity_result.to_csv(LIQUIDITY_CSV, index=False)
        final_tickers = liquidity_result[liquidity_result["crit6_liquidity"]]["ticker"].tolist()

    print(f"[screen] {len(final_tickers)} tickers pass all 6 criteria: {final_tickers}")

    print(f"[screen] Building {CHART_LOOKBACK_DAYS}-day chart data for the dashboard ...")
    chart_data = build_chart_data(daily, final_tickers)

    # Merge summary stats (from criteria_result) and liquidity stats for
    # each final passer, so the dashboard has key numbers to display
    # alongside each chart without needing to re-derive them client-side.
    summary_cols = [
        "ticker", "close", "sma50", "sma150", "sma200",
        "low_52w", "high_52w", "ret_1m", "spx_ret_1m", "ret_1w", "spx_ret_1w",
    ]
    summary_lookup = (
        criteria_result[criteria_result["ticker"].isin(final_tickers)][summary_cols]
        .set_index("ticker")
        .to_dict(orient="index")
    )
    liquidity_lookup = (
        liquidity_result[liquidity_result["ticker"].isin(final_tickers)][
            ["ticker", "market_cap", "avg_volume_3m"]
        ]
        .set_index("ticker")
        .to_dict(orient="index")
        if not liquidity_result.empty else {}
    )

    stocks_out = []
    for ticker in final_tickers:
        stats = summary_lookup.get(ticker, {})
        liq = liquidity_lookup.get(ticker, {})
        stocks_out.append({
            "ticker": ticker,
            "close": stats.get("close"),
            "sma50": stats.get("sma50"),
            "sma150": stats.get("sma150"),
            "sma200": stats.get("sma200"),
            "low_52w": stats.get("low_52w"),
            "high_52w": stats.get("high_52w"),
            "ret_1m": stats.get("ret_1m"),
            "spx_ret_1m": stats.get("spx_ret_1m"),
            "ret_1w": stats.get("ret_1w"),
            "spx_ret_1w": stats.get("spx_ret_1w"),
            "market_cap": liq.get("market_cap"),
            "avg_volume_3m": liq.get("avg_volume_3m"),
            "chart": chart_data.get(ticker, []),
        })

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe_size": int(daily["ticker"].nunique()),
        "criteria_1_4_pass_count": len(survivors),
        "vcp_pass_count": len(vcp_passers),
        "final_pass_count": len(final_tickers),
        "min_market_cap": MIN_MARKET_CAP,
        "min_avg_volume_3m": MIN_AVG_VOLUME,
        "spx_ret_1m": spx_ret_1m,
        "spx_ret_1w": spx_ret_1w,
        "stocks": stocks_out,
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"[screen] Wrote final results (with chart data) to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
