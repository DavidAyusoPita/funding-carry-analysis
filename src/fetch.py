"""Download public funding-rate and price history from four perpetual venues.

Only public REST endpoints are used - no API keys, no authentication. Every
response is cached under ``data/`` so the notebook can be re-run offline and
the analysis stays reproducible against a fixed snapshot.

The four venues were chosen to span the two market structures that matter for
this analysis: three centralised order-book exchanges (Binance, Bybit, OKX)
and one on-chain perpetual DEX (Hyperliquid). The cross-venue funding spread
between those two groups is the subject of the analysis.

A note on units, because it is the single most dangerous detail here: venues
publish funding over different intervals. Binance, Bybit and OKX quote a rate
per 8-hour period; Hyperliquid quotes a rate per hour. Comparing the raw
numbers overstates the CEX rates by 8x. Everything returned by this module is
therefore normalised to a **fraction per hour**, and the interval is inferred
from the data rather than hard-coded, because venues do change it (Binance
moves volatile symbols to a 4-hour schedule).
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

BINANCE_FUTURES = "https://fapi.binance.com"
BYBIT = "https://api.bybit.com"
OKX = "https://www.okx.com"
HYPERLIQUID = "https://api.hyperliquid.xyz"

HOURS_PER_YEAR = 24 * 365

# Symbol naming differs per venue; this keeps the notebook free of string surgery.
SYMBOL_MAP = {
    "BTC": {"binance": "BTCUSDT", "bybit": "BTCUSDT", "okx": "BTC-USDT-SWAP", "hyperliquid": "BTC"},
    "ETH": {"binance": "ETHUSDT", "bybit": "ETHUSDT", "okx": "ETH-USDT-SWAP", "hyperliquid": "ETH"},
    "SOL": {"binance": "SOLUSDT", "bybit": "SOLUSDT", "okx": "SOL-USDT-SWAP", "hyperliquid": "SOL"},
}

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "funding-carry-analysis/1.0"})


# --------------------------------------------------------------------------- #
# Caching helpers
# --------------------------------------------------------------------------- #

def _cache_path(name: str) -> Path:
    return DATA_DIR / f"{name}.csv"


def _load_cache(name: str) -> pd.DataFrame | None:
    path = _cache_path(name)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # ISO8601 rather than a single format string: some venues stamp settlements
    # a millisecond past the hour, so the cached strings are not uniform.
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    return df


def _save_cache(df: pd.DataFrame, name: str) -> None:
    df.to_csv(_cache_path(name), index=False)


def _get(url: str, **kwargs) -> dict | list:
    resp = _SESSION.get(url, timeout=25, **kwargs)
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------- #
# Unit normalisation
# --------------------------------------------------------------------------- #

def infer_interval_hours(times: pd.Series) -> float:
    """Funding interval implied by the timestamps, in hours.

    Inferred rather than assumed: Binance quotes most symbols on an 8-hour
    schedule but moves volatile ones to 4 hours, and a hard-coded 8 would
    silently halve those rates. The median gap is robust to the occasional
    missing settlement.
    """
    if len(times) < 3:
        return 8.0
    gap_hours = times.sort_values().diff().dt.total_seconds().median() / 3600
    # Snap to the schedules venues actually use, so noise does not produce 7.98.
    return min((1.0, 2.0, 4.0, 8.0), key=lambda c: abs(c - gap_hours))


def _normalise(df: pd.DataFrame, venue: str) -> pd.DataFrame:
    """Attach the inferred interval and a per-hour rate to a raw funding frame."""
    out = df.copy()
    # Binance stamps settlements a millisecond past the hour; flooring keeps the
    # timestamps joinable across venues and the inferred interval clean.
    out["time"] = out["time"].dt.floor("s")
    interval = infer_interval_hours(out["time"])
    out["venue"] = venue
    out["interval_hours"] = interval
    out["rate_per_hour"] = out["funding_rate"] / interval
    out["annualised_pct"] = out["rate_per_hour"] * HOURS_PER_YEAR * 100
    return out


# --------------------------------------------------------------------------- #
# Per-venue funding history
# --------------------------------------------------------------------------- #

def binance_funding(coin: str = "BTC", pages: int = 6, use_cache: bool = True) -> pd.DataFrame:
    """Funding history from Binance USD-M perpetuals (1000 records per page)."""
    symbol = SYMBOL_MAP[coin]["binance"]
    name = f"funding_binance_{coin}"
    if use_cache and (cached := _load_cache(name)) is not None:
        return cached

    rows: list[dict] = []
    end_time: int | None = None
    for _ in range(pages):
        params: dict = {"symbol": symbol, "limit": 1000}
        if end_time is not None:
            params["endTime"] = end_time
        batch = _get(f"{BINANCE_FUTURES}/fapi/v1/fundingRate", params=params)
        if not batch:
            break
        rows.extend(batch)
        end_time = int(batch[0]["fundingTime"]) - 1
        time.sleep(0.25)

    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    df = pd.DataFrame({
        "time": pd.to_datetime(raw["fundingTime"], unit="ms", utc=True),
        "funding_rate": raw["fundingRate"].astype(float),
    }).drop_duplicates("time").sort_values("time").reset_index(drop=True)

    df = _normalise(df, "binance")
    _save_cache(df, name)
    return df


def bybit_funding(coin: str = "BTC", pages: int = 25, use_cache: bool = True) -> pd.DataFrame:
    """Funding history from Bybit linear perpetuals (200 records per page)."""
    symbol = SYMBOL_MAP[coin]["bybit"]
    name = f"funding_bybit_{coin}"
    if use_cache and (cached := _load_cache(name)) is not None:
        return cached

    rows: list[dict] = []
    end_time: int | None = None
    for _ in range(pages):
        params: dict = {"category": "linear", "symbol": symbol, "limit": 200}
        if end_time is not None:
            params["endTime"] = end_time
        batch = _get(f"{BYBIT}/v5/market/funding/history", params=params)
        batch = batch.get("result", {}).get("list", [])
        if not batch:
            break
        rows.extend(batch)
        end_time = min(int(r["fundingRateTimestamp"]) for r in batch) - 1
        time.sleep(0.25)

    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    df = pd.DataFrame({
        "time": pd.to_datetime(raw["fundingRateTimestamp"].astype("int64"), unit="ms", utc=True),
        "funding_rate": raw["fundingRate"].astype(float),
    }).drop_duplicates("time").sort_values("time").reset_index(drop=True)

    df = _normalise(df, "bybit")
    _save_cache(df, name)
    return df


def okx_funding(coin: str = "BTC", pages: int = 50, use_cache: bool = True) -> pd.DataFrame:
    """Funding history from OKX swaps (100 records per page).

    OKX caps this endpoint at roughly three months of history regardless of how
    far the pagination is pushed, so OKX constrains any window that includes it.
    """
    inst = SYMBOL_MAP[coin]["okx"]
    name = f"funding_okx_{coin}"
    if use_cache and (cached := _load_cache(name)) is not None:
        return cached

    rows: list[dict] = []
    after: int | None = None
    for _ in range(pages):
        params: dict = {"instId": inst, "limit": 100}
        if after is not None:
            params["after"] = after
        batch = _get(f"{OKX}/api/v5/public/funding-rate-history", params=params)
        batch = batch.get("data", [])
        if not batch:
            break
        rows.extend(batch)
        after = min(int(r["fundingTime"]) for r in batch)
        time.sleep(0.2)

    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    df = pd.DataFrame({
        "time": pd.to_datetime(raw["fundingTime"].astype("int64"), unit="ms", utc=True),
        "funding_rate": raw["realizedRate"].astype(float),
    }).drop_duplicates("time").sort_values("time").reset_index(drop=True)

    df = _normalise(df, "okx")
    _save_cache(df, name)
    return df


def hyperliquid_funding(coin: str = "BTC", start: str = "2024-07-01",
                        use_cache: bool = True, max_pages: int = 60) -> pd.DataFrame:
    """Funding history from Hyperliquid, the on-chain venue (hourly, 500/page).

    Paginates forward from ``start``; the endpoint returns records in ascending
    time order beginning at ``startTime``.
    """
    hl_coin = SYMBOL_MAP[coin]["hyperliquid"]
    name = f"funding_hyperliquid_{coin}"
    if use_cache and (cached := _load_cache(name)) is not None:
        return cached

    cursor = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    rows: list[dict] = []
    for _ in range(max_pages):
        payload = {"type": "fundingHistory", "coin": hl_coin, "startTime": cursor}
        batch = _SESSION.post(f"{HYPERLIQUID}/info", json=payload, timeout=25).json()
        if not batch:
            break
        rows.extend(batch)
        newest = max(int(r["time"]) for r in batch)
        if newest <= cursor:
            break
        cursor = newest + 1
        time.sleep(0.2)

    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    df = pd.DataFrame({
        "time": pd.to_datetime(raw["time"].astype("int64"), unit="ms", utc=True),
        "funding_rate": raw["fundingRate"].astype(float),
    }).drop_duplicates("time").sort_values("time").reset_index(drop=True)

    df = _normalise(df, "hyperliquid")
    _save_cache(df, name)
    return df


VENUE_FETCHERS = {
    "binance": binance_funding,
    "bybit": bybit_funding,
    "okx": okx_funding,
    "hyperliquid": hyperliquid_funding,
}


def all_venues(coin: str = "BTC", use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Funding history for one coin across every venue, normalised to per-hour."""
    out = {}
    for venue, fetcher in VENUE_FETCHERS.items():
        try:
            df = fetcher(coin=coin, use_cache=use_cache)
        except Exception as exc:                              # noqa: BLE001
            print(f"  {venue:<12} {coin}: failed ({exc})")
            continue
        if df.empty:
            print(f"  {venue:<12} {coin}: no data")
            continue
        out[venue] = df
        print(f"  {venue:<12} {coin}: {len(df):>5} points, "
              f"{df['interval_hours'].iloc[0]:.0f}h interval, "
              f"from {df['time'].min():%Y-%m-%d}")
    return out


# --------------------------------------------------------------------------- #
# Prices - used to measure the cross-venue basis, which is a real entry cost
# --------------------------------------------------------------------------- #

def binance_perp_prices(coin: str = "BTC", interval: str = "1h", pages: int = 12,
                        use_cache: bool = True) -> pd.DataFrame:
    """Hourly perpetual closes from Binance."""
    symbol = SYMBOL_MAP[coin]["binance"]
    name = f"price_binance_{coin}_{interval}"
    if use_cache and (cached := _load_cache(name)) is not None:
        return cached

    rows: list[list] = []
    end_time: int | None = None
    for _ in range(pages):
        params: dict = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end_time is not None:
            params["endTime"] = end_time
        batch = _get(f"{BINANCE_FUTURES}/fapi/v1/klines", params=params)
        if not batch:
            break
        rows.extend(batch)
        end_time = int(batch[0][0]) - 1
        time.sleep(0.25)

    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows).iloc[:, :5]
    raw.columns = ["open_time", "open", "high", "low", "close"]
    df = pd.DataFrame({
        "time": pd.to_datetime(raw["open_time"], unit="ms", utc=True),
        "close": raw["close"].astype(float),
    }).drop_duplicates("time").sort_values("time").reset_index(drop=True)

    _save_cache(df, name)
    return df


def hyperliquid_prices(coin: str = "BTC", start: str = "2025-07-01",
                       use_cache: bool = True, max_pages: int = 40) -> pd.DataFrame:
    """Hourly perpetual closes from Hyperliquid (5000 candles per request)."""
    hl_coin = SYMBOL_MAP[coin]["hyperliquid"]
    name = f"price_hyperliquid_{coin}_1h"
    if use_cache and (cached := _load_cache(name)) is not None:
        return cached

    cursor = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    rows: list[dict] = []
    for _ in range(max_pages):
        payload = {"type": "candleSnapshot",
                   "req": {"coin": hl_coin, "interval": "1h",
                           "startTime": cursor, "endTime": now_ms}}
        batch = _SESSION.post(f"{HYPERLIQUID}/info", json=payload, timeout=25).json()
        if not batch:
            break
        rows.extend(batch)
        newest = max(int(r["t"]) for r in batch)
        if newest <= cursor:
            break
        cursor = newest + 1
        time.sleep(0.2)

    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    df = pd.DataFrame({
        "time": pd.to_datetime(raw["t"].astype("int64"), unit="ms", utc=True),
        "close": raw["c"].astype(float),
    }).drop_duplicates("time").sort_values("time").reset_index(drop=True)

    _save_cache(df, name)
    return df
