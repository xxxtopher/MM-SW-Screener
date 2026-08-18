"""
screen.py

Orchestrates the full screening pipeline:
  1. Load data/daily_prices/latest.parquet
  2. Run criteria 1-4c (criteria.py) across the whole universe - includes
     margin-based relative strength gates on 1-month, 2-week, and 3-month
     horizons, plus an alpha_score for every ticker
  3. Run criterion 5 / VCP (vcp.py) only on the criteria survivors
  4. Run criterion 6 / liquidity gate (liquidity.py) only on the VCP
     survivors
  5. Rank all survivors by alpha_score and keep only the top TOP_N - a
     stable-sized, quality-ranked final list rather than "everyone who
     happens to clear the hard gates today" (added 2026-08-10)
  6. For the top N, embed the last ~3 months of daily OHLC (for the
     dashboard's mini price charts) directly into the output JSON
  7. Write output/screen_results.json

This assumes data/daily_prices/latest.parquet is already fresh (i.e.
fetch_prices.py has already been run).

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
TOP_N = 30                 # show only the top N survivors, ranked by alpha_score


# ---------------------------------------------------------------------------
# Chart data
# ---------------------------------------------------------------------------

def build_chart_data(daily_df: pd.DataFrame, tickers: list[str]) -> dict[str, list[dict]]:
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

    print("[screen] Fetching SPX 1-month, 2-week, and 3-month return benchmarks ...")
    spx_returns = fetch_spx_returns()
    spx_ret_1m = spx_returns["ret_1m"]
    spx_ret_2w = spx_returns["ret_2w"]
    spx_ret_3m = spx_returns["ret_3m"]
    print(f"[screen] SPX 1-month return: {spx_ret_1m:.2%}")
    print(f"[screen] SPX 2-week return: {spx_ret_2w:.2%}")
    print(f"[screen] SPX 3-month return: {spx_ret_3m:.2%}")

    print("[screen] Running criteria 1-4c across the full universe ...")
    criteria_result = compute_criteria(
        daily, spx_ret_1m=spx_ret_1m, spx_ret_2w=spx_ret_2w, spx_ret_3m=spx_ret_3m
    )
    criteria_result.to_csv(CRITERIA_CSV, index=False)

    survivors = criteria_result[criteria_result["pass_all"]]["ticker"].tolist()
    print(f"[screen] {len(survivors)} of {len(criteria_result)} pass criteria 1-4c")

    if not survivors:
        print("[screen] No survivors from criteria 1-4c - writing empty result set.")
        vcp_passers: list[str] = []
        liquidity_result = pd.DataFrame(columns=["ticker", "market_cap", "avg_volume_3m", "crit6_liquidity"])
    else:
        print(f"[screen] Running VCP check (criterion 5) on {len(survivors)} survivors ...")
        vcp_result = compute_vcp(daily, survivors)
        vcp_result.to_csv(VCP_CSV, index=False)
        vcp_passers = vcp_result[vcp_result["crit5_vcp"]]["ticker"].tolist()

    print(f"[screen] {len(vcp_passers)} tickers pass criteria 1-5 (before liquidity gate)")

    if not vcp_passers:
        print("[screen] No VCP survivors - skipping liquidity check.")
        liquidity_result = pd.DataFrame(columns=["ticker", "market_cap", "avg_volume_3m", "crit6_liquidity"])
        gated_tickers: list[str] = []
    else:
        print(
            f"[screen] Checking liquidity (market cap >= ${MIN_MARKET_CAP:,.0f}, "
            f"3-month avg volume >= {MIN_AVG_VOLUME:,}) on {len(vcp_passers)} survivors ..."
        )
        liquidity_result = compute_liquidity(daily, vcp_passers)
        liquidity_result.to_csv(LIQUIDITY_CSV, index=False)
        gated_tickers = liquidity_result[liquidity_result["crit6_liquidity"]]["ticker"].tolist()

    print(f"[screen] {len(gated_tickers)} tickers pass all 6 hard-gate criteria")

    # Rank all gate-passers by alpha_score and keep only the top N.
    alpha_lookup = criteria_result.set_index("ticker")["alpha_score"].to_dict()
    ranked_tickers = sorted(gated_tickers, key=lambda t: alpha_lookup.get(t, float("-inf")), reverse=True)
    final_tickers = ranked_tickers[:TOP_N]

    print(f"[screen] Showing top {len(final_tickers)} of {len(gated_tickers)} by alpha_score: {final_tickers}")

    print(f"[screen] Building {CHART_LOOKBACK_DAYS}-day chart data for the dashboard ...")
    chart_data = build_chart_data(daily, final_tickers)

    summary_cols = [
        "ticker", "close", "sma20", "sma50", "sma150", "sma200",
        "low_52w", "high_52w",
        "ret_1m", "spx_ret_1m", "excess_1m",
        "ret_2w", "spx_ret_2w", "excess_2w",
        "ret_3m", "spx_ret_3m", "excess_3m",
        "alpha_score",
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
    for rank, ticker in enumerate(final_tickers, start=1):
        stats = summary_lookup.get(ticker, {})
        liq = liquidity_lookup.get(ticker, {})
        stocks_out.append({
            "rank": rank,
            "ticker": ticker,
            "close": stats.get("close"),
            "sma20": stats.get("sma20"),
            "sma50": stats.get("sma50"),
            "sma150": stats.get("sma150"),
            "sma200": stats.get("sma200"),
            "low_52w": stats.get("low_52w"),
            "high_52w": stats.get("high_52w"),
            "ret_1m": stats.get("ret_1m"),
            "spx_ret_1m": stats.get("spx_ret_1m"),
            "ret_2w": stats.get("ret_2w"),
            "spx_ret_2w": stats.get("spx_ret_2w"),
            "ret_3m": stats.get("ret_3m"),
            "spx_ret_3m": stats.get("spx_ret_3m"),
            "alpha_score": stats.get("alpha_score"),
            "market_cap": liq.get("market_cap"),
            "avg_volume_3m": liq.get("avg_volume_3m"),
            "chart": chart_data.get(ticker, []),
        })

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe_size": int(daily["ticker"].nunique()),
        "criteria_1_4c_pass_count": len(survivors),
        "vcp_pass_count": len(vcp_passers),
        "gated_pass_count": len(gated_tickers),
        "final_pass_count": len(final_tickers),
        "top_n": TOP_N,
        "min_market_cap": MIN_MARKET_CAP,
        "min_avg_volume_3m": MIN_AVG_VOLUME,
        "spx_ret_1m": spx_ret_1m,
        "spx_ret_2w": spx_ret_2w,
        "spx_ret_3m": spx_ret_3m,
        "stocks": stocks_out,
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"[screen] Wrote final results (with chart data) to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
