"""
IV (Initiation Volume) Strategy
================================
Logic:
  1. Consolidation: (Highest High - Lowest Low) / Lowest Low <= 10% over 15+ trading days
  2. IV Candle: Green candle (close > open) with:
       - Body size (close - open) / open >= 2%
       - Volume > 30-day average AND largest in last 30 days
  3. Entry: First daily close above IV candle high within 15 trading days -> enter at market close
  4. Stop Loss: max(IV candle low, entry_price * 0.85)  -> whichever is HIGHER price-wise
  5. Target: entry_price * 1.30  (30% gain)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------
# Data Classes
# ---------------------------------------------

@dataclass
class IVCandle:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    body_pct: float            # candle body size as % of open
    consol_start: str
    consol_days: int
    consol_range_pct: float


@dataclass
class IVSignal:
    ticker: str
    status: str                # "ENTRY_TRIGGERED" | "WATCHING" | "EXPIRED" | "NO_SIGNAL"

    iv_candle: Optional[IVCandle] = None
    entry_date: Optional[str] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    stop_loss_basis: Optional[str] = None
    target: Optional[float] = None
    risk_reward: Optional[float] = None
    days_since_iv: Optional[int] = None
    days_remaining: Optional[int] = None
    message: str = ""
    price_history: list = field(default_factory=list)


# ---------------------------------------------
# Core Strategy Engine
# ---------------------------------------------

class IVStrategy:

    CONSOL_MIN_DAYS   = 15
    CONSOL_MAX_RANGE  = 0.10    # 10%
    VOL_AVG_WINDOW    = 30
    ENTRY_WINDOW_DAYS = 15
    TARGET_PCT        = 0.30
    SL_PCT            = 0.15
    IV_MIN_BODY_PCT   = 0.02    # IV candle body must be >= 2% of open price

    def __init__(self, df: pd.DataFrame, ticker: str):
        self.ticker = ticker
        self.df = self._prepare(df)

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        df["date"]       = pd.to_datetime(df["date"])
        df               = df.sort_values("date").reset_index(drop=True)
        df["vol_30_avg"] = df["volume"].rolling(self.VOL_AVG_WINDOW).mean()
        df["vol_30_max"] = df["volume"].rolling(self.VOL_AVG_WINDOW).max()
        df["is_green"]   = df["close"] > df["open"]
        df["body_pct"]   = (df["close"] - df["open"]) / df["open"]
        return df

    def _find_consolidation(self, up_to_idx: int) -> Optional[dict]:
        end_idx = up_to_idx
        best    = None
        hi      = self.df.loc[end_idx, "high"]
        lo      = self.df.loc[end_idx, "low"]

        for start_idx in range(end_idx, -1, -1):
            hi        = max(hi, self.df.loc[start_idx, "high"])
            lo        = min(lo, self.df.loc[start_idx, "low"])
            range_pct = (hi - lo) / lo

            if range_pct > self.CONSOL_MAX_RANGE:
                break

            days = end_idx - start_idx + 1
            if days >= self.CONSOL_MIN_DAYS:
                best = {
                    "start_idx": start_idx,
                    "end_idx":   end_idx,
                    "days":      days,
                    "range_pct": range_pct,
                    "hi":        hi,
                    "lo":        lo,
                }

        return best

    def _is_iv_candle(self, idx: int) -> bool:
        row = self.df.loc[idx]

        # Must be green
        if not row["is_green"]:
            return False

        # Need valid rolling stats
        if pd.isna(row["vol_30_avg"]) or pd.isna(row["vol_30_max"]):
            return False

        # NEW: body must be >= 2% of open
        strong_body = row["body_pct"] >= self.IV_MIN_BODY_PCT

        # Volume above 30-day avg AND largest in 30 days
        above_avg  = row["volume"] > row["vol_30_avg"]
        largest_30 = row["volume"] >= row["vol_30_max"]

        return strong_body and above_avg and largest_30

    def analyze(self) -> IVSignal:
        df = self.df
        n  = len(df)

        if n < self.VOL_AVG_WINDOW + self.CONSOL_MIN_DAYS:
            return IVSignal(
                ticker=self.ticker, status="NO_SIGNAL",
                message="Not enough historical data (need 45+ trading days)."
            )

        iv_idx      = None
        consol_info = None
        search_start = max(self.VOL_AVG_WINDOW, n - 60)

        for idx in range(n - 1, search_start - 1, -1):
            if not self._is_iv_candle(idx):
                continue
            c = self._find_consolidation(idx - 1) if idx > 0 else None
            if c is None:
                continue
            iv_idx      = idx
            consol_info = c
            break

        if iv_idx is None:
            return IVSignal(
                ticker=self.ticker, status="NO_SIGNAL",
                message="No IV candle found in the last 60 trading days. Conditions: green candle with body >= 2%, highest 30-day volume, preceded by 15+ day consolidation within 10% range.",
                price_history=self._price_history(n - 60, n - 1)
            )

        iv_row   = df.loc[iv_idx]
        body_pct = round(float(iv_row["body_pct"]) * 100, 2)

        iv_candle = IVCandle(
            date             = str(iv_row["date"].date()),
            open             = round(float(iv_row["open"]),  2),
            high             = round(float(iv_row["high"]),  2),
            low              = round(float(iv_row["low"]),   2),
            close            = round(float(iv_row["close"]), 2),
            volume           = int(iv_row["volume"]),
            body_pct         = body_pct,
            consol_start     = str(df.loc[consol_info["start_idx"], "date"].date()),
            consol_days      = consol_info["days"],
            consol_range_pct = round(consol_info["range_pct"] * 100, 2),
        )

        watch_start    = iv_idx + 1
        watch_end      = min(iv_idx + self.ENTRY_WINDOW_DAYS, n - 1)
        days_since_iv  = n - 1 - iv_idx
        days_remaining = max(0, self.ENTRY_WINDOW_DAYS - days_since_iv)

        entry_idx = None
        for i in range(watch_start, watch_end + 1):
            if df.loc[i, "close"] > iv_candle.high:
                entry_idx = i
                break

        if entry_idx is None and days_since_iv > self.ENTRY_WINDOW_DAYS:
            return IVSignal(
                ticker=self.ticker, status="EXPIRED",
                iv_candle=iv_candle,
                days_since_iv=days_since_iv, days_remaining=0,
                message=f"IV candle on {iv_candle.date} (body: {body_pct}%). Entry window expired — no close above ${iv_candle.high}.",
                price_history=self._price_history(consol_info["start_idx"], n - 1)
            )

        if entry_idx is None:
            return IVSignal(
                ticker=self.ticker, status="WATCHING",
                iv_candle=iv_candle,
                days_since_iv=days_since_iv, days_remaining=days_remaining,
                message=(
                    f"IV candle on {iv_candle.date} — body: {body_pct}%, "
                    f"watching for close above ${iv_candle.high:.2f}. "
                    f"{days_remaining} trading day(s) left in entry window."
                ),
                price_history=self._price_history(consol_info["start_idx"], n - 1)
            )

        entry_row   = df.loc[entry_idx]
        entry_price = round(float(entry_row["close"]), 2)

        sl_iv_low = iv_candle.low
        sl_15pct  = round(entry_price * (1 - self.SL_PCT), 2)
        if sl_iv_low >= sl_15pct:
            stop_loss = sl_iv_low
            sl_basis  = "IV_LOW"
        else:
            stop_loss = sl_15pct
            sl_basis  = "15_PCT"

        target = round(entry_price * (1 + self.TARGET_PCT), 2)
        risk   = entry_price - stop_loss
        reward = target - entry_price
        rr     = round(reward / risk, 2) if risk > 0 else None

        return IVSignal(
            ticker=self.ticker, status="ENTRY_TRIGGERED",
            iv_candle=iv_candle,
            entry_date=str(entry_row["date"].date()),
            entry_price=entry_price,
            stop_loss=round(stop_loss, 2),
            stop_loss_basis=sl_basis,
            target=target,
            risk_reward=rr,
            days_since_iv=days_since_iv, days_remaining=0,
            message=(
                f"Entry triggered on {entry_row['date'].date()} at ${entry_price}. "
                f"IV candle body: {body_pct}%. "
                f"Target ${target} | SL ${round(stop_loss, 2)} ({sl_basis})."
            ),
            price_history=self._price_history(consol_info["start_idx"], n - 1)
        )

    def _price_history(self, start_idx: int, end_idx: int) -> list:
        rows = self.df.loc[start_idx:end_idx, ["date", "open", "high", "low", "close", "volume"]]
        result = []
        for _, r in rows.iterrows():
            result.append({
                "date":   str(r["date"].date()),
                "open":   round(float(r["open"]),  2),
                "high":   round(float(r["high"]),  2),
                "low":    round(float(r["low"]),   2),
                "close":  round(float(r["close"]), 2),
                "volume": int(r["volume"]),
            })
        return result


# ---------------------------------------------
# Entry point called by app.py
# ---------------------------------------------

def run_iv_strategy(df: pd.DataFrame, ticker: str) -> dict:
    strat  = IVStrategy(df, ticker)
    signal = strat.analyze()

    out = {
        "ticker":          signal.ticker,
        "status":          signal.status,
        "message":         signal.message,
        "price_history":   signal.price_history,
        "iv_candle":       None,
        "entry":           None,
        "risk_management": None,
    }

    if signal.iv_candle:
        out["iv_candle"] = {
            "date":             signal.iv_candle.date,
            "open":             signal.iv_candle.open,
            "high":             signal.iv_candle.high,
            "low":              signal.iv_candle.low,
            "close":            signal.iv_candle.close,
            "volume":           signal.iv_candle.volume,
            "body_pct":         signal.iv_candle.body_pct,
            "consol_start":     signal.iv_candle.consol_start,
            "consol_days":      signal.iv_candle.consol_days,
            "consol_range_pct": signal.iv_candle.consol_range_pct,
        }

    if signal.entry_price:
        out["entry"] = {
            "date":  signal.entry_date,
            "price": signal.entry_price,
        }
        out["risk_management"] = {
            "stop_loss":       signal.stop_loss,
            "stop_loss_basis": signal.stop_loss_basis,
            "target":          signal.target,
            "risk_reward":     signal.risk_reward,
        }

    if signal.days_since_iv is not None:
        out["days_since_iv"]  = signal.days_since_iv
        out["days_remaining"] = signal.days_remaining

    return out
