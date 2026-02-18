"""
backtest.py — Fast vectorized backtester for IV Strategy & Fusion 5
====================================================================
Computes all indicators once upfront using pandas/numpy.
No per-bar strategy re-instantiation. Runs in seconds not minutes.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


# ─────────────────────────────────────────────
# Trade record
# ─────────────────────────────────────────────

@dataclass
class Trade:
    strategy:      str
    entry_date:    str
    entry_price:   float
    sl:            float
    target:        float
    target2:       Optional[float] = None
    exit_date:     Optional[str]   = None
    exit_price:    Optional[float] = None
    exit_reason:   Optional[str]   = None  # TARGET | TARGET1 | TARGET2 | SL | TIMEOUT
    return_pct:    Optional[float] = None
    duration_days: Optional[int]   = None


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def _simulate_forward(df: pd.DataFrame, entry_idx: int,
                      entry_price: float, sl: float,
                      target: float, target2: Optional[float],
                      strategy: str, max_hold: int = 60) -> Trade:
    """Simulate a single trade forward from entry_idx+1."""
    entry_date = str(df.loc[entry_idx, "date"].date())
    n = len(df)

    for j in range(entry_idx + 1, min(entry_idx + 1 + max_hold, n)):
        lo  = float(df.loc[j, "low"])
        hi  = float(df.loc[j, "high"])
        dt  = str(df.loc[j, "date"].date())
        dur = j - entry_idx

        if lo <= sl:
            ret = round((sl - entry_price) / entry_price * 100, 2)
            return Trade(strategy, entry_date, entry_price, sl, target, target2,
                         dt, sl, "SL", ret, dur)

        if target2 and hi >= target2:
            ret = round((target2 - entry_price) / entry_price * 100, 2)
            return Trade(strategy, entry_date, entry_price, sl, target, target2,
                         dt, target2, "TARGET2", ret, dur)

        if hi >= target:
            ret = round((target - entry_price) / entry_price * 100, 2)
            label = "TARGET1" if target2 else "TARGET"
            return Trade(strategy, entry_date, entry_price, sl, target, target2,
                         dt, target, label, ret, dur)

    # Timeout — exit at last available close
    last_idx   = min(entry_idx + max_hold, n - 1)
    last_close = float(df.loc[last_idx, "close"])
    last_date  = str(df.loc[last_idx, "date"].date())
    ret        = round((last_close - entry_price) / entry_price * 100, 2)
    return Trade(strategy, entry_date, entry_price, sl, target, target2,
                 last_date, round(last_close, 2), "TIMEOUT", ret, max_hold)


# ─────────────────────────────────────────────
# IV Strategy — vectorized signal detection
# ─────────────────────────────────────────────

def _compute_iv_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add all IV columns to df in one pass."""
    df = df.copy()

    # Rolling volume stats
    df["vol_avg30"] = df["volume"].rolling(30).mean()
    df["vol_max30"] = df["volume"].rolling(30).max()

    # Green candle & body
    df["is_green"]  = df["close"] > df["open"]
    df["body_pct"]  = (df["close"] - df["open"]) / df["open"]

    # IV candle flag
    df["is_iv"] = (
        df["is_green"] &
        (df["body_pct"] >= 0.02) &
        (df["volume"] > df["vol_avg30"]) &
        (df["volume"] >= df["vol_max30"])
    )

    return df


def _find_consol_before(df: pd.DataFrame, end_idx: int,
                        min_days: int = 15, max_range: float = 0.10) -> Optional[dict]:
    """
    Vectorized consolidation check ending at end_idx.
    Returns dict or None.
    """
    if end_idx < min_days:
        return None

    # Expand backwards until range breaks 10%
    hi = df.loc[end_idx, "high"]
    lo = df.loc[end_idx, "low"]
    start = end_idx

    for i in range(end_idx, max(end_idx - 120, -1), -1):
        hi = max(hi, df.loc[i, "high"])
        lo = min(lo, df.loc[i, "low"])
        if (hi - lo) / lo > max_range:
            break
        days = end_idx - i + 1
        if days >= min_days:
            start = i

    days = end_idx - start + 1
    if days < min_days:
        return None

    return {
        "start_idx":  start,
        "end_idx":    end_idx,
        "range_low":  float(df.loc[start:end_idx, "low"].min()),
    }


def run_iv_backtest(df: pd.DataFrame, ticker: str,
                    lookback_days: int = 504) -> List[Trade]:
    """
    Fast IV Strategy backtest.
    lookback_days: how many bars of the df tail to scan (504 ≈ 2 yrs of trading days)
    """
    df    = df.copy().reset_index(drop=True)
    df    = _compute_iv_signals(df)
    n     = len(df)
    start = max(0, n - lookback_days)

    trades    = []
    skip_until = start  # skip bars while in a trade

    for i in range(start, n):
        if i < skip_until:
            continue
        if not df.loc[i, "is_iv"]:
            continue

        # Need consolidation before this candle
        consol = _find_consol_before(df, i - 1)
        if consol is None:
            continue

        iv_high = float(df.loc[i, "high"])
        iv_low  = float(df.loc[i, "low"])

        # Watch next 15 bars for first close above IV high
        entry_idx = None
        for j in range(i + 1, min(i + 16, n)):
            if float(df.loc[j, "close"]) > iv_high:
                entry_idx = j
                break

        if entry_idx is None:
            continue  # no trigger in window

        entry_price = round(float(df.loc[entry_idx, "close"]), 2)
        sl_iv       = iv_low
        sl_15pct    = round(entry_price * 0.85, 2)
        sl          = max(sl_iv, sl_15pct)
        risk        = entry_price - sl          # $ risk per share
        target      = round(entry_price + 2 * risk, 2)   # 2R target

        trade = _simulate_forward(df, entry_idx, entry_price,
                                  round(sl, 2), target, None, "IV Strategy")
        trades.append(trade)

        # Don't take another trade until this one exits
        if trade.exit_date:
            exit_rows = df[df["date"].dt.strftime("%Y-%m-%d") == trade.exit_date]
            if not exit_rows.empty:
                skip_until = exit_rows.index[0] + 1
                continue
        skip_until = entry_idx + 61

    return trades


# ─────────────────────────────────────────────
# Fusion 5 — vectorized signal detection
# ─────────────────────────────────────────────

def _compute_fusion5_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 5 EMAs + BB + fusion flag in one pass."""
    df = df.copy()

    for p in [4, 9, 18, 50, 200]:
        df[f"ema{p}"] = df["close"].ewm(span=p, adjust=False).mean()

    df["bb_mid"]   = df["close"].rolling(20).mean()
    df["bb_std"]   = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]

    # EMA spread at each bar
    ema_cols = ["ema4", "ema9", "ema18", "ema50", "ema200"]
    df["ema_hi"]     = df[ema_cols].max(axis=1)
    df["ema_lo"]     = df[ema_cols].min(axis=1)
    df["ema_spread"] = (df["ema_hi"] - df["ema_lo"]) / df["ema_lo"]

    # Fusion flag: spread between 1% and 2.5%
    df["in_fusion"] = (df["ema_spread"] >= 0.01) & (df["ema_spread"] <= 0.025)

    # BB breakout flag
    df["bb_breakout"] = df["close"] > df["bb_upper"]

    return df


def run_fusion5_backtest(df: pd.DataFrame, ticker: str,
                         lookback_days: int = 504) -> List[Trade]:
    """Fast Fusion 5 backtest."""
    df    = df.copy().reset_index(drop=True)
    df    = _compute_fusion5_signals(df)
    n     = len(df)
    start = max(230, n - lookback_days)  # need 230 bars for 200 EMA warmup

    trades     = []
    skip_until = start

    i = start
    while i < n:
        if i < skip_until:
            i += 1
            continue

        # Look for a fusion zone ending at or before i
        # Find how long the current or most-recent fusion run was
        if not df.loc[i, "in_fusion"]:
            i += 1
            continue

        # Walk back to find start of this fusion run
        zone_end = i
        zone_start = i
        for k in range(i, max(i - 120, -1), -1):
            if df.loc[k, "in_fusion"]:
                zone_start = k
            else:
                break

        zone_days = zone_end - zone_start + 1
        if zone_days < 3:
            i += 1
            continue

        # range_low of fusion zone
        range_low = float(df.loc[zone_start:zone_end, "low"].min())

        # Look for BB breakout after fusion zone
        entry_idx = None
        for j in range(zone_end + 1, min(zone_end + 30, n)):
            if df.loc[j, "bb_breakout"] and not pd.isna(df.loc[j, "bb_upper"]):
                entry_idx = j
                break

        if entry_idx is None:
            i = zone_end + 1
            continue

        entry_price = round(float(df.loc[entry_idx, "high"]), 2)  # enter at candle high
        sl_range    = round(range_low * 0.985, 2)
        sl_12pct    = round(entry_price * 0.88, 2)
        sl          = max(sl_range, sl_12pct)
        risk        = entry_price - sl          # $ risk per share
        target      = round(entry_price + 2 * risk, 2)   # 2R target

        trade = _simulate_forward(df, entry_idx, entry_price,
                                  round(sl, 2), target, None, "Fusion 5")
        trades.append(trade)

        if trade.exit_date:
            exit_rows = df[df["date"].dt.strftime("%Y-%m-%d") == trade.exit_date]
            if not exit_rows.empty:
                skip_until = exit_rows.index[0] + 1
                i = skip_until
                continue
        i = entry_idx + 61


    return trades


# ─────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────

def compute_stats(trades: List[Trade]) -> dict:
    completed = [t for t in trades if t.return_pct is not None]
    if not completed:
        return {"total": 0, "message": "No trades triggered in this period."}

    returns  = [t.return_pct for t in completed]
    wins     = [r for r in returns if r > 0]
    losses   = [r for r in returns if r <= 0]

    win_rate   = round(len(wins) / len(completed) * 100, 1)
    avg_return = round(float(np.mean(returns)), 2)
    avg_win    = round(float(np.mean(wins)),  2) if wins   else 0.0
    avg_loss   = round(float(np.mean(losses)),2) if losses else 0.0
    best       = round(max(returns), 2)
    worst      = round(min(returns), 2)

    cum  = np.cumsum(returns)
    peak = np.maximum.accumulate(cum)
    dd   = cum - peak
    max_dd = round(float(np.min(dd)), 2)

    gross_wins   = sum(w for w in wins)
    gross_losses = abs(sum(l for l in losses))
    pf = round(gross_wins / gross_losses, 2) if gross_losses > 0 else 9999.0

    reasons = {}
    for t in completed:
        r = t.exit_reason or "OPEN"
        reasons[r] = reasons.get(r, 0) + 1

    return {
        "total":         len(completed),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      win_rate,
        "avg_return":    avg_return,
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "best_trade":    best,
        "worst_trade":   worst,
        "max_drawdown":  max_dd,
        "profit_factor": pf,
        "exit_reasons":  reasons,
    }


def _trade_to_dict(t: Trade) -> dict:
    return {
        "strategy":      t.strategy,
        "entry_date":    t.entry_date,
        "entry_price":   t.entry_price,
        "exit_date":     t.exit_date,
        "exit_price":    t.exit_price,
        "exit_reason":   t.exit_reason,
        "return_pct":    t.return_pct,
        "duration_days": t.duration_days,
        "sl":            round(t.sl, 2),
        "target":        round(t.target, 2),
        "target2":       round(t.target2, 2) if t.target2 else None,
    }


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, ticker: str, strategy: str) -> dict:
    """strategy: 'iv' | 'fusion5' | 'both'"""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    results = {}

    if strategy in ("iv", "both"):
        trades = run_iv_backtest(df, ticker)
        results["iv"] = {
            "stats":  compute_stats(trades),
            "trades": [_trade_to_dict(t) for t in trades],
        }

    if strategy in ("fusion5", "both"):
        trades = run_fusion5_backtest(df, ticker)
        results["fusion5"] = {
            "stats":  compute_stats(trades),
            "trades": [_trade_to_dict(t) for t in trades],
        }

    return results
