"""
Fusion 5 — Convergence System (Strategy 2)
===========================================
Logic:
  1. FUSION: 4 EMAs (4, 9, 18, 50) converge within 1%-2.5% spread
             200 EMA is NOT part of the fusion — used only as trend filter
             on the daily chart for 3+ consecutive days
  2. ENTRY:  After fusion, first candle that closes above Upper Bollinger Band
             (20 SMA, 2 std dev) → enter at HIGH of that breakout candle
  3. SL:     Low of convergence range * 0.985 (−1.5%) OR entry * 0.88 (−12%)
             whichever is HIGHER
  4. T1:     entry * 1.20  (20%)
     T2:     entry * 1.30  (30%)
  5. EXIT:   Alert if price closes below 50 EMA after entry
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class FusionZone:
    start_date: str
    end_date: str
    days: int
    spread_pct: float        # actual EMA spread % during fusion
    ema4:  float
    ema9:  float
    ema18: float
    ema50: float
    range_low: float
    ema200_now: float        # 200 EMA at entry — trend filter only


@dataclass
class Fusion5Signal:
    ticker: str
    status: str              # ENTRY_TRIGGERED | WATCHING | NO_SIGNAL

    fusion_zone: Optional[FusionZone] = None

    # Entry
    entry_date: Optional[str] = None
    entry_price: Optional[float] = None   # high of breakout candle

    # Breakout candle details
    breakout_close: Optional[float] = None
    upper_band: Optional[float] = None

    # Risk management
    stop_loss: Optional[float] = None
    stop_loss_basis: Optional[str] = None   # "RANGE_LOW" or "12_PCT"
    target1: Optional[float] = None         # 20%
    target2: Optional[float] = None         # 30%

    # 50 EMA exit warning
    below_50ema: Optional[bool] = None
    ema50_current: Optional[float] = None

    message: str = ""
    price_history: list = field(default_factory=list)


# ─────────────────────────────────────────────
# Core Strategy Engine
# ─────────────────────────────────────────────

class Fusion5Strategy:

    # EMAs — 4,9,18,50 used for convergence; 200 used as above/below filter only
    EMA_PERIODS        = [4, 9, 18, 50, 200]
    FUSION_EMA_PERIODS = [4, 9, 18, 50]

    # Fusion thresholds
    FUSION_MIN_SPREAD = 0.010   # 1.0%
    FUSION_MAX_SPREAD = 0.025   # 2.5%
    FUSION_MIN_DAYS   = 3

    # Bollinger Bands
    BB_PERIOD = 20
    BB_STD    = 2.0

    # Risk management
    SL_BUFFER  = 0.015   # 1.5% below range low
    SL_MAX_PCT = 0.12    # 12% max SL

    # Targets
    T1_PCT = 0.20
    T2_PCT = 0.30

    def __init__(self, df: pd.DataFrame, ticker: str):
        self.ticker = ticker
        self.df = self._prepare(df)

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # EMAs
        for p in self.EMA_PERIODS:
            df[f"ema{p}"] = df["close"].ewm(span=p, adjust=False).mean()

        # Bollinger Bands (20 SMA ± 2 std)
        df["bb_mid"]   = df["close"].rolling(self.BB_PERIOD).mean()
        df["bb_std"]   = df["close"].rolling(self.BB_PERIOD).std()
        df["bb_upper"] = df["bb_mid"] + self.BB_STD * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - self.BB_STD * df["bb_std"]

        return df

    # ── Check if a single row is in fusion ──
    def _spread_at(self, idx: int) -> Optional[float]:
        """Spread across 4 convergence EMAs only (4, 9, 18, 50). 200 EMA excluded."""
        row = self.df.loc[idx]
        emas = [row[f"ema{p}"] for p in self.FUSION_EMA_PERIODS]
        if any(pd.isna(v) for v in emas):
            return None
        hi = max(emas)
        lo = min(emas)
        return (hi - lo) / lo

    # ── Find the most recent fusion zone ending just before search_end ──
    def _find_fusion_zone(self, search_end: int) -> Optional[dict]:
        """
        Walk backwards from search_end to find the latest block of 3+
        consecutive days where EMA spread is between 1% and 2.5%.
        """
        # Find end of most recent qualifying run
        end_idx = None
        for i in range(search_end, -1, -1):
            s = self._spread_at(i)
            if s is None:
                continue
            if self.FUSION_MIN_SPREAD <= s <= self.FUSION_MAX_SPREAD:
                end_idx = i
                break

        if end_idx is None:
            return None

        # Expand backwards
        start_idx = end_idx
        for i in range(end_idx - 1, -1, -1):
            s = self._spread_at(i)
            if s is not None and self.FUSION_MIN_SPREAD <= s <= self.FUSION_MAX_SPREAD:
                start_idx = i
            else:
                break

        days = end_idx - start_idx + 1
        if days < self.FUSION_MIN_DAYS:
            return None

        # Gather zone stats
        zone_rows  = self.df.loc[start_idx:end_idx]
        spread_avg = float(np.mean([self._spread_at(i) for i in range(start_idx, end_idx + 1)
                                    if self._spread_at(i) is not None]))
        last_row   = self.df.loc[end_idx]

        return {
            "start_idx":  start_idx,
            "end_idx":    end_idx,
            "days":       days,
            "spread_pct": round(spread_avg * 100, 2),
            "range_low":  float(zone_rows["low"].min()),
            "ema4":       round(float(last_row["ema4"]),   2),
            "ema9":       round(float(last_row["ema9"]),   2),
            "ema18":      round(float(last_row["ema18"]),  2),
            "ema50":      round(float(last_row["ema50"]),  2),
        }

    # ── Main analysis ──
    def analyze(self) -> Fusion5Signal:
        df = self.df
        n  = len(df)

        MIN_BARS = max(self.EMA_PERIODS) + self.BB_PERIOD + 10  # 200 EMA needs 200 bars
        if n < MIN_BARS:
            return Fusion5Signal(
                ticker=self.ticker, status="NO_SIGNAL",
                message=f"Not enough data (need {MIN_BARS}+ trading days)."
            )

        # ── Look back up to 90 days for a fusion zone ──
        search_start = max(0, n - 90)

        # Find fusion zone ending anywhere in the look-back window
        fusion_info = None
        for end_candidate in range(n - 1, search_start, -1):
            s = self._spread_at(end_candidate)
            if s is None or not (self.FUSION_MIN_SPREAD <= s <= self.FUSION_MAX_SPREAD):
                continue
            info = self._find_fusion_zone(end_candidate)
            if info:
                fusion_info = info
                break

        if fusion_info is None:
            return Fusion5Signal(
                ticker=self.ticker, status="NO_SIGNAL",
                message=(
                    "No Fusion Zone found in the last 90 days. "
                    "Need 4 EMAs (4/9/18/50) within 1%–2.5% spread for 3+ days, then BB breakout with price above 200 EMA."
                ),
                price_history=self._price_history(max(0, n - 90), n - 1)
            )

        fusion_zone = FusionZone(
            start_date  = str(df.loc[fusion_info["start_idx"], "date"].date()),
            end_date    = str(df.loc[fusion_info["end_idx"],   "date"].date()),
            days        = fusion_info["days"],
            spread_pct  = fusion_info["spread_pct"],
            ema4        = fusion_info["ema4"],
            ema9        = fusion_info["ema9"],
            ema18       = fusion_info["ema18"],
            ema50       = fusion_info["ema50"],
            range_low   = round(fusion_info["range_low"], 2),
            ema200_now  = round(float(df.loc[n-1, "ema200"]), 2),
        )

        # ── Scan for BB breakout AFTER fusion zone ──
        watch_start = fusion_info["end_idx"] + 1
        watch_end   = n - 1

        entry_idx = None
        for i in range(watch_start, watch_end + 1):
            row = df.loc[i]
            if pd.isna(row["bb_upper"]) or pd.isna(row["ema200"]):
                continue
            # Entry: close above BB upper AND price above 200 EMA
            if row["close"] > row["bb_upper"] and row["close"] > row["ema200"]:
                entry_idx = i
                break

        # Current 50 EMA
        ema50_now = round(float(df.loc[n - 1, "ema50"]), 2)
        last_close = float(df.loc[n - 1, "close"])
        below_50   = last_close < ema50_now

        # ── Still watching — fusion found, no breakout yet ──
        if entry_idx is None:
            return Fusion5Signal(
                ticker       = self.ticker,
                status       = "WATCHING",
                fusion_zone  = fusion_zone,
                below_50ema  = below_50,
                ema50_current= ema50_now,
                message=(
                    f"Fusion Zone detected ({fusion_zone.start_date} → {fusion_zone.end_date}, "
                    f"{fusion_zone.days} days, spread {fusion_zone.spread_pct}%). "
                    f"Watching for close above Upper Bollinger Band while price stays above 200 EMA (${round(float(df.loc[n-1, 'ema200']), 2)})."
                ),
                price_history=self._price_history(fusion_info["start_idx"], n - 1)
            )

        # ── Entry triggered ──
        entry_row      = df.loc[entry_idx]
        entry_price    = round(float(entry_row["high"]), 2)   # enter at HIGH of breakout candle
        breakout_close = round(float(entry_row["close"]), 2)
        upper_band     = round(float(entry_row["bb_upper"]), 2)

        # SL: range_low * (1 - 1.5%)  OR  entry * (1 - 12%)  → higher price wins
        sl_range = round(fusion_info["range_low"] * (1 - self.SL_BUFFER), 2)
        sl_12pct = round(entry_price * (1 - self.SL_MAX_PCT), 2)

        if sl_range >= sl_12pct:
            stop_loss = sl_range
            sl_basis  = "RANGE_LOW"
        else:
            stop_loss = sl_12pct
            sl_basis  = "12_PCT"

        target1 = round(entry_price * (1 + self.T1_PCT), 2)
        target2 = round(entry_price * (1 + self.T2_PCT), 2)

        # 50 EMA exit warning (based on latest close)
        below_50_now = last_close < ema50_now

        return Fusion5Signal(
            ticker          = self.ticker,
            status          = "ENTRY_TRIGGERED",
            fusion_zone     = fusion_zone,
            entry_date      = str(entry_row["date"].date()),
            entry_price     = entry_price,
            breakout_close  = breakout_close,
            upper_band      = upper_band,
            stop_loss       = stop_loss,
            stop_loss_basis = sl_basis,
            target1         = target1,
            target2         = target2,
            below_50ema     = below_50_now,
            ema50_current   = ema50_now,
            message=(
                f"BB Breakout on {entry_row['date'].date()}. "
                f"Entry at candle high ${entry_price}. "
                f"T1 ${target1} (+20%) | T2 ${target2} (+30%) | SL ${stop_loss} ({sl_basis})."
                + (" ⚠ Price now below 50 EMA — monitor exit." if below_50_now else "")
            ),
            price_history=self._price_history(fusion_info["start_idx"], n - 1)
        )

    def _price_history(self, start_idx: int, end_idx: int) -> list:
        cols = ["date", "open", "high", "low", "close", "volume",
                "ema4", "ema9", "ema18", "ema50", "ema200",
                "bb_upper", "bb_mid", "bb_lower"]
        rows = self.df.loc[start_idx:end_idx, cols]
        result = []
        for _, r in rows.iterrows():
            result.append({
                "date":      str(r["date"].date()),
                "open":      round(float(r["open"]),  2),
                "high":      round(float(r["high"]),  2),
                "low":       round(float(r["low"]),   2),
                "close":     round(float(r["close"]), 2),
                "volume":    int(r["volume"]),
                "ema4":      round(float(r["ema4"]),   2) if not pd.isna(r["ema4"])   else None,
                "ema9":      round(float(r["ema9"]),   2) if not pd.isna(r["ema9"])   else None,
                "ema18":     round(float(r["ema18"]),  2) if not pd.isna(r["ema18"])  else None,
                "ema50":     round(float(r["ema50"]),  2) if not pd.isna(r["ema50"])  else None,
                "ema200":    round(float(r["ema200"]), 2) if not pd.isna(r["ema200"]) else None,
                "bb_upper":  round(float(r["bb_upper"]),2) if not pd.isna(r["bb_upper"]) else None,
                "bb_mid":    round(float(r["bb_mid"]),  2) if not pd.isna(r["bb_mid"])   else None,
                "bb_lower":  round(float(r["bb_lower"]),2) if not pd.isna(r["bb_lower"]) else None,
            })
        return result


# ─────────────────────────────────────────────
# Entry point called by app.py
# ─────────────────────────────────────────────

def run_fusion5_strategy(df: pd.DataFrame, ticker: str) -> dict:
    strat  = Fusion5Strategy(df, ticker)
    signal = strat.analyze()

    out = {
        "ticker":        signal.ticker,
        "status":        signal.status,
        "message":       signal.message,
        "price_history": signal.price_history,
        "fusion_zone":   None,
        "entry":         None,
        "risk_management": None,
        "exit_warning":  signal.below_50ema or False,
        "ema50_current": signal.ema50_current,
    }

    if signal.fusion_zone:
        fz = signal.fusion_zone
        out["fusion_zone"] = {
            "start_date":  fz.start_date,
            "end_date":    fz.end_date,
            "days":        fz.days,
            "spread_pct":  fz.spread_pct,
            "range_low":   fz.range_low,
            "emas": {
                "ema4":   fz.ema4,
                "ema9":   fz.ema9,
                "ema18":  fz.ema18,
                "ema50":  fz.ema50,
            },
            "ema200_now": fz.ema200_now,
        }

    if signal.entry_price:
        out["entry"] = {
            "date":           signal.entry_date,
            "price":          signal.entry_price,
            "breakout_close": signal.breakout_close,
            "upper_band":     signal.upper_band,
        }
        out["risk_management"] = {
            "stop_loss":       signal.stop_loss,
            "stop_loss_basis": signal.stop_loss_basis,
            "target1":         signal.target1,
            "target2":         signal.target2,
        }

    return out
