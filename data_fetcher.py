"""
data_fetcher.py — Fetch OHLCV data without yfinance
Uses Yahoo Finance's direct CSV download URL — works on cloud servers
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://finance.yahoo.com/",
}


def _get_crumb(session: requests.Session) -> str:
    """Get Yahoo Finance crumb token needed for API calls"""
    # Step 1: visit Yahoo Finance to get cookies
    session.get("https://fc.yahoo.com", headers=HEADERS, timeout=10)
    # Step 2: get crumb
    resp = session.get(
        "https://query1.finance.yahoo.com/v1/test/getcrumb",
        headers=HEADERS,
        timeout=10
    )
    crumb = resp.text.strip()
    return crumb


def fetch_data(ticker: str, days: int = 500) -> pd.DataFrame:
    """Fetch daily OHLCV using Yahoo Finance v8 API with crumb auth"""

    end_ts   = int(datetime.today().timestamp())
    start_ts = int((datetime.today() - timedelta(days=days)).timestamp())

    session = requests.Session()
    session.headers.update(HEADERS)

    errors = []

    # Method 1: Yahoo Finance v8 with crumb
    try:
        crumb = _get_crumb(session)
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?period1={start_ts}&period2={end_ts}"
            f"&interval=1d&events=history&crumb={crumb}"
        )
        resp = session.get(url, headers=HEADERS, timeout=15)
        data = resp.json()

        result = data.get("chart", {}).get("result", [])
        if result:
            r          = result[0]
            timestamps = r["timestamp"]
            quotes     = r["indicators"]["quote"][0]
            adjclose   = r["indicators"].get("adjclose", [{}])[0].get("adjclose", quotes["close"])

            df = pd.DataFrame({
                "date":   pd.to_datetime(timestamps, unit="s"),
                "open":   quotes["open"],
                "high":   quotes["high"],
                "low":    quotes["low"],
                "close":  adjclose,
                "volume": quotes["volume"],
            })
            df = df.dropna().sort_values("date").reset_index(drop=True)
            if len(df) > 10:
                return df
    except Exception as e:
        errors.append(f"v8-crumb: {e}")

    # Method 2: Yahoo Finance v7 download (CSV)
    try:
        url = (
            f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
            f"?period1={start_ts}&period2={end_ts}"
            f"&interval=1d&events=history"
        )
        resp = session.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200 and "Date" in resp.text[:50]:
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={"adj close": "close"})
            df["date"] = pd.to_datetime(df["date"])
            df = df[["date", "open", "high", "low", "close", "volume"]]
            df = df.dropna().sort_values("date").reset_index(drop=True)
            if len(df) > 10:
                return df
    except Exception as e:
        errors.append(f"v7-csv: {e}")

    # Method 3: query2 fallback
    try:
        url = (
            f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?period1={start_ts}&period2={end_ts}&interval=1d"
        )
        resp = session.get(url, headers=HEADERS, timeout=15)
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if result:
            r          = result[0]
            timestamps = r["timestamp"]
            quotes     = r["indicators"]["quote"][0]
            adjclose   = r["indicators"].get("adjclose", [{}])[0].get("adjclose", quotes["close"])
            df = pd.DataFrame({
                "date":   pd.to_datetime(timestamps, unit="s"),
                "open":   quotes["open"],
                "high":   quotes["high"],
                "low":    quotes["low"],
                "close":  adjclose,
                "volume": quotes["volume"],
            })
            df = df.dropna().sort_values("date").reset_index(drop=True)
            if len(df) > 10:
                return df
    except Exception as e:
        errors.append(f"query2: {e}")

    raise ValueError(
        f"Could not fetch data for '{ticker}'. "
        f"Try again in a moment. ({'; '.join(errors)})"
    )


def current_price(ticker: str):
    """Get current price from Yahoo Finance"""
    try:
        session = requests.Session()
        session.headers.update(HEADERS)
        crumb = _get_crumb(session)
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?range=1d&interval=1m&crumb={crumb}"
        )
        resp = session.get(url, headers=HEADERS, timeout=10)
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if result:
            meta = result[0].get("meta", {})
            price = meta.get("regularMarketPrice") or meta.get("previousClose")
            if price:
                return round(float(price), 2)
    except Exception:
        pass
    return None
