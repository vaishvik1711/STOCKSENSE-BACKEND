"""
app.py — StockSense AI Backend
Strategies: IV Strategy + Fusion 5
Auto-scans both strategies and returns combined results.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import traceback
import os
import json
import base64

from iv_strategy      import run_iv_strategy
from fusion5_strategy import run_fusion5_strategy
from backtest         import run_backtest
from excel_export     import build_excel

app = Flask(__name__)
CORS(app)


# ─────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────

def fetch_data(ticker: str, days: int = 500) -> pd.DataFrame:
    """
    500 calendar days ~ 350 trading days.
    Fusion 5 needs 200 EMA + 20 BB + 3 fusion days = ~230 trading days minimum.
    """
    end   = datetime.today()
    start = end - timedelta(days=days)
    raw   = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True
    )

    if raw.empty:
        raise ValueError(f"No data returned for '{ticker}'. Check the symbol.")

    raw = raw.reset_index()
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    raw = raw.rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume"
    })
    return raw[["date", "open", "high", "low", "close", "volume"]]


def current_price(ticker: str):
    try:
        return round(float(yf.Ticker(ticker).fast_info.last_price), 2)
    except Exception:
        return None


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/api/scan", methods=["POST"])
def scan():
    """
    Main endpoint — runs BOTH strategies on the ticker and returns combined result.
    The frontend shows whichever strategy has an active signal.
    """
    body   = request.get_json(silent=True) or {}
    ticker = body.get("ticker", "").strip().upper()

    if not ticker:
        return jsonify({"error": "ticker is required"}), 400

    try:
        df    = fetch_data(ticker)
        price = current_price(ticker)
        asof  = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        iv_result      = run_iv_strategy(df, ticker)
        fusion5_result = run_fusion5_strategy(df, ticker)

        iv_result["current_price"]      = price
        iv_result["as_of"]              = asof
        fusion5_result["current_price"] = price
        fusion5_result["as_of"]         = asof

        return jsonify({
            "ticker":  ticker,
            "price":   price,
            "as_of":   asof,
            "iv":      iv_result,
            "fusion5": fusion5_result,
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal server error. Check terminal for details."}), 500


# Keep old endpoints working too
@app.route("/api/analyze", methods=["POST"])
def analyze():
    body   = request.get_json(silent=True) or {}
    ticker = body.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400
    try:
        df     = fetch_data(ticker)
        result = run_iv_strategy(df, ticker)
        result["current_price"] = current_price(ticker)
        result["as_of"]         = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal server error."}), 500


@app.route("/api/fusion5", methods=["POST"])
def fusion5():
    body   = request.get_json(silent=True) or {}
    ticker = body.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400
    try:
        df     = fetch_data(ticker, days=500)
        result = run_fusion5_strategy(df, ticker)
        result["current_price"] = current_price(ticker)
        result["as_of"]         = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal server error."}), 500



@app.route("/api/backtest", methods=["POST"])
def backtest():
    """
    Vectorized backtest over 2 years of data. Runs in seconds.
    Body: { "ticker": "AAPL", "strategy": "iv" | "fusion5" | "both" }
    """
    body     = request.get_json(silent=True) or {}
    ticker   = body.get("ticker",   "").strip().upper()
    strategy = body.get("strategy", "both").strip().lower()

    if not ticker:
        return jsonify({"error": "ticker is required"}), 400
    if strategy not in ("iv", "fusion5", "both"):
        return jsonify({"error": "strategy must be iv, fusion5, or both"}), 400

    try:
        # 900 calendar days = ~630 trading days (2 yrs data + 200 EMA warmup)
        df     = fetch_data(ticker, days=900)
        result = run_backtest(df, ticker, strategy)
        result["ticker"]   = ticker
        result["strategy"] = strategy
        result["as_of"]    = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        result["period"]   = "2 years"
        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal server error. Check terminal for details."}), 500



@app.route("/api/watchlist", methods=["POST"])
def watchlist():
    """
    Stream watchlist results one ticker at a time using SSE.
    Body: { "tickers": ["AAPL","NVDA",...], "strategy": "both"|"iv"|"fusion5" }
    """
    body     = request.get_json(silent=True) or {}
    tickers  = body.get("tickers", [])
    strategy = body.get("strategy", "both").strip().lower()

    # Validate
    tickers = [t.strip().upper() for t in tickers if t.strip()][:20]
    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400

    def generate():
        all_results = []

        for i, ticker in enumerate(tickers):
            try:
                # Fetch data once — used for both scan and backtest
                df = fetch_data(ticker, days=900)
                cp = current_price(ticker)
                asof = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

                # Live scan
                iv_result  = run_iv_strategy(df, ticker)
                f5_result  = run_fusion5_strategy(df, ticker)
                iv_result["current_price"]  = cp
                f5_result["current_price"]  = cp

                # Backtest
                bt_result = run_backtest(df, ticker, strategy)

                result = {
                    "ticker":   ticker,
                    "price":    cp,
                    "as_of":   asof,
                    "index":   i,
                    "total":   len(tickers),
                    "scan":    {"iv": iv_result, "fusion5": f5_result},
                    "backtest": bt_result,
                }
                all_results.append(result)

                # Stream this ticker's result
                payload = json.dumps({"type": "ticker", "data": result})
                yield f"data: {payload}\n\n"

            except Exception as e:
                traceback.print_exc()
                err_payload = json.dumps({
                    "type": "error",
                    "ticker": ticker,
                    "message": str(e)
                })
                yield f"data: {err_payload}\n\n"

        # Build and stream Excel as base64
        try:
            xlsx_bytes  = build_excel(all_results)
            xlsx_b64    = base64.b64encode(xlsx_bytes).decode("utf-8")
            done_payload = json.dumps({
                "type":  "done",
                "total": len(all_results),
                "excel": xlsx_b64,
            })
            yield f"data: {done_payload}\n\n"
        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'type':'done','total':len(all_results),'excel':None})}\n\n"

    return app.response_class(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering":"no",
            "Connection":       "keep-alive",
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"\n🚀  StockSense AI running on port {port}")
    print(f"    Strategies: IV Strategy · Fusion 5")
    print(f"    Endpoints: /api/scan · /api/backtest\n")
    app.run(host="0.0.0.0", port=port, debug=True)
