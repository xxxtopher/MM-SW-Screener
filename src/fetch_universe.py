"""
fetch_universe.py

Downloads the current holdings of the iShares Russell 3000 ETF (IWV) and
caches a clean ticker universe to data/universe.csv.

This is meant to run on a LOW FREQUENCY schedule (e.g. monthly), separate
from the daily screener. The daily screener reads data/universe.csv only
and never calls out to iShares directly.

If the iShares download fails (endpoint changed, blocked, etc.), this
script falls back to scraping S&P 500 + S&P 400 + S&P 600 constituent
lists from Wikipedia as a smaller but reliable substitute universe.

Usage:
    python src/fetch_universe.py
"""

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = REPO_ROOT / "data" / "universe.csv"
META_JSON = REPO_ROOT / "data" / "universe_meta.json"

# IWV = iShares Russell 3000 ETF, portfolioId=239714
ISHARES_URL = (
    "https://www.blackrock.com/varnish-api/blk-one01-product-data/"
    "product-data/api/v1/get-fund-document"
    "?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares"
    "&locale=en_US&portfolioId=239714&component=fundDownload&userType=individual"
)

HEADERS = {
    # Some vendor endpoints reject requests with no browser-like UA.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
WIKI_SP600_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"


# ---------------------------------------------------------------------------
# Primary source: iShares IWV holdings CSV
# ---------------------------------------------------------------------------

def fetch_ishares_holdings() -> pd.DataFrame:
    """
    Downloads the IWV holdings file and returns a cleaned DataFrame with
    columns: ticker, name, sector, asset_class, weight_pct

    Raises on any failure so the caller can fall back to the Wikipedia route.
    """
    resp = requests.get(ISHARES_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    raw_text = resp.content.decode("utf-8-sig", errors="replace")
    lines = raw_text.splitlines()

    # The iShares holdings CSV has several metadata/disclaimer rows before
    # the actual holdings table. The real header row starts with "Ticker".
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Ticker,"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            "Could not locate holdings header row in iShares CSV "
            "(file format may have changed)."
        )

    csv_body = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_body), thousands=",")

    # Normalize column names (iShares sometimes tweaks casing/spacing).
    df.columns = [c.strip().lower().replace(" ", "_").replace("(%)", "pct") for c in df.columns]

    rename_map = {
        "ticker": "ticker",
        "name": "name",
        "sector": "sector",
        "asset_class": "asset_class",
        "weight_pct": "weight_pct",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = {"ticker", "name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"iShares CSV missing expected columns: {missing}")

    if "asset_class" in df.columns:
        df = df[df["asset_class"].astype(str).str.strip().str.lower() == "equity"]

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    # Drop rows with no real ticker (cash, futures, blanks, "-")
    df = df[df["ticker"].notna()]
    df = df[~df["ticker"].isin(["", "-", "NAN", "N/A"])]

    # Drop tickers containing characters that indicate non-standard listings
    # (e.g. rights/warrants "XYZ.WS", foreign ords) - keep it simple/clean
    # for a yfinance-driven screener. Adjust this filter later if needed.
    df = df[df["ticker"].str.match(r"^[A-Z]{1,6}$")]

    keep_cols = [c for c in ["ticker", "name", "sector", "weight_pct"] if c in df.columns]
    df = df[keep_cols].drop_duplicates(subset="ticker").reset_index(drop=True)

    if len(df) < 1000:
        # Sanity check - a real IWV pull should have ~2,000-2,700 equities.
        raise ValueError(
            f"Parsed only {len(df)} tickers from iShares CSV - "
            "this looks wrong, treating as a failed fetch."
        )

    return df


# ---------------------------------------------------------------------------
# Fallback source: Wikipedia S&P 500 + 400 + 600
# ---------------------------------------------------------------------------

def fetch_wikipedia_sp_universe() -> pd.DataFrame:
    """
    Fallback universe: S&P 500 + S&P 400 (mid cap) + S&P 600 (small cap),
    scraped from Wikipedia. Smaller than the true Russell 3000 but reliable
    and dependency-free.
    """
    frames = []
    for url, symbol_col_guess in [
        (WIKI_SP500_URL, "Symbol"),
        (WIKI_SP400_URL, "Symbol"),
        (WIKI_SP600_URL, "Symbol"),
    ]:
        tables = pd.read_html(url)
        # The constituents table is typically the first table on each page.
        table = tables[0]
        table.columns = [str(c).strip() for c in table.columns]

        symbol_col = symbol_col_guess if symbol_col_guess in table.columns else table.columns[0]
        name_col = "Security" if "Security" in table.columns else (
            "Company" if "Company" in table.columns else table.columns[1]
        )
        sector_col = "GICS Sector" if "GICS Sector" in table.columns else None

        sub = pd.DataFrame({
            "ticker": table[symbol_col].astype(str).str.strip().str.upper(),
            "name": table[name_col].astype(str).str.strip(),
        })
        if sector_col:
            sub["sector"] = table[sector_col].astype(str).str.strip()

        frames.append(sub)

    df = pd.concat(frames, ignore_index=True)
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B for yfinance
    df = df[df["ticker"].str.match(r"^[A-Z\-]{1,6}$")]
    df = df.drop_duplicates(subset="ticker").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    source_used = None
    df = None
    error_detail = None

    try:
        df = fetch_ishares_holdings()
        source_used = "ishares_iwv"
        print(f"[fetch_universe] iShares IWV fetch succeeded: {len(df)} tickers")
    except Exception as exc:  # noqa: BLE001 - want to fall back on anything
        error_detail = str(exc)
        print(f"[fetch_universe] iShares IWV fetch failed: {exc}", file=sys.stderr)
        print("[fetch_universe] Falling back to Wikipedia S&P 500+400+600 ...", file=sys.stderr)
        df = fetch_wikipedia_sp_universe()
        source_used = "wikipedia_sp500_400_600_fallback"
        print(f"[fetch_universe] Fallback fetch succeeded: {len(df)} tickers")

    df.to_csv(OUTPUT_CSV, index=False)

    meta = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_used": source_used,
        "ticker_count": len(df),
        "primary_source_error": error_detail,
    }
    META_JSON.write_text(json.dumps(meta, indent=2))

    print(f"[fetch_universe] Saved {len(df)} tickers to {OUTPUT_CSV}")
    print(f"[fetch_universe] Source: {source_used}")


if __name__ == "__main__":
    main()
