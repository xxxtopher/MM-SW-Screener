"""
fetch_prices.py

Batch-downloads ~1 year of daily OHLCV data for every ticker in
data/universe.csv, using yfinance, and caches the result as a single
Parquet file per run date: data/daily_prices/prices_YYYY-MM-DD.parquet

Also writes data/daily_prices/latest.parquet (always the most recent pull)
so downstream scripts (criteria.py, vcp.py, screen.py) have one stable
path to read from without needing to know today's date.

Failures (delisted tickers, no data returned, etc.) are logged to
data/daily_prices/fetch_failures_YYYY-MM-DD.csv rather than crashing the
whole run.

Usage:
    python src/fetch_prices.py
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_CSV = REPO_ROOT / "data" / "universe.csv"
PRICES_DIR = REPO_ROOT / "data" / "daily_prices"

PERIOD = "1y"          # ~250 trading days - covers 200-day SMA + buffer
INTERVAL = "1d"
BATCH_SIZE = 150        # tickers per yf.download() call
BATCH_PAUSE_SEC = 2.0   # pause between batches to avoid rate limiting
DOWNLOAD_THREADS = True

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_universe() -> list[str]:
    if not UNIVERSE_CSV.exists():
        raise FileNotFoundError(
            f"{UNIVERSE_CSV} not found - run fetch_universe.py first."
        )
    df = pd.read_csv(UNIVERSE_CSV)
    tickers = df["ticker"].dropna().astype(str).str.strip().str.upper().unique().tolist()
    return sorted(tickers)


def chunk(lst: list[str], size: int) -> list[list[str]]:
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def download_batch(tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Downloads one batch of tickers. Returns (long_format_df, failed_tickers).
    long_format_df columns: ticker, date, open, high, low, close, volume
    """
    raw = yf.download(
        tickers=tickers,
        period=PERIOD,
        interval=INTERVAL,
        group_by="ticker",
        threads=DOWNLOAD_THREADS,
        auto_adjust=True,   # adjusted close folded into 'Close' - simpler downstream
        progress=False,
    )

    frames = []
    failed = []

    # Single-ticker batches return a flat (non-multiindex) frame from yfinance.
    is_multi = isinstance(raw.columns, pd.MultiIndex)

    for t in tickers:
        try:
            if is_multi:
                if t not in raw.columns.get_level_values(0):
                    failed.append(t)
                    continue
                sub = raw[t].copy()
            else:
                sub = raw.copy()

            sub = sub.dropna(subset=["Close"])
            if sub.empty or not set(REQUIRED_COLUMNS).issubset(sub.columns):
                failed.append(t)
                continue

            sub = sub.reset_index()
            sub["ticker"] = t
            sub = sub.rename(columns={
                "Date": "date", "Open": "open", "High": "high",
                "Low": "low", "Close": "close", "Volume": "volume",
            })
            frames.append(sub[["ticker", "date", "open", "high", "low", "close", "volume"]])
        except Exception:
            failed.append(t)

    if frames:
        return pd.concat(frames, ignore_index=True), failed
    return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"]), failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    PRICES_DIR.mkdir(parents=True, exist_ok=True)

    tickers = load_universe()
    print(f"[fetch_prices] Loaded {len(tickers)} tickers from universe.csv")

    batches = chunk(tickers, BATCH_SIZE)
    print(f"[fetch_prices] Downloading in {len(batches)} batches of up to {BATCH_SIZE} tickers")

    all_frames = []
    all_failed: list[str] = []

    for i, batch in enumerate(batches, start=1):
        print(f"[fetch_prices] Batch {i}/{len(batches)} ({len(batch)} tickers)...")
        try:
            df_batch, failed_batch = download_batch(batch)
        except Exception as exc:  # noqa: BLE001 - keep the run alive
            print(f"[fetch_prices] Batch {i} failed entirely: {exc}")
            failed_batch = batch
            df_batch = pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])

        all_frames.append(df_batch)
        all_failed.extend(failed_batch)

        if failed_batch:
            print(f"[fetch_prices]   -> {len(failed_batch)} tickers failed in this batch")

        if i < len(batches):
            time.sleep(BATCH_PAUSE_SEC)

    combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

    if combined.empty:
        raise RuntimeError(
            "[fetch_prices] No price data was downloaded at all - aborting "
            "before writing empty output."
        )

    combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dated_path = PRICES_DIR / f"prices_{today_str}.parquet"
    latest_path = PRICES_DIR / "latest.parquet"

    combined.to_parquet(dated_path, index=False)
    combined.to_parquet(latest_path, index=False)

    n_ok = combined["ticker"].nunique()
    print(f"[fetch_prices] Saved {len(combined):,} rows across {n_ok} tickers")
    print(f"[fetch_prices]   -> {dated_path}")
    print(f"[fetch_prices]   -> {latest_path}")

    if all_failed:
        failures_path = PRICES_DIR / f"fetch_failures_{today_str}.csv"
        pd.DataFrame({"ticker": sorted(set(all_failed))}).to_csv(failures_path, index=False)
        print(f"[fetch_prices] {len(set(all_failed))} tickers failed - logged to {failures_path}")


if __name__ == "__main__":
    main()
