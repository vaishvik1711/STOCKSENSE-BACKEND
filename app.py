"""
app.py — StockSense AI Backend
Strategies: IV Strategy + Fusion 5
"""

from flask import Flask, request, jsonify, Response, stream_with_context
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
CORS(app, resources={r"/api/*": {
    "origins": ["*"],
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
    "supports_credentials": False
}})

@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin",  "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response


# ─────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────

def fetch_data(ticker: str, days: int = 500) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=days)

    # yfinance gets blocked on cloud IPs — use session with browser headers
    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })

    raw = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
        session=session,
    )
    if raw.empty:
        # Retry once without session in case headers cause issues
        raw = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
    if raw.empty:
        raise ValueError(f"No data for '{ticker}'. Check the symbol or try again in a moment.")

    raw = raw.reset_index()
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    raw = raw.rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume"
    })
    return raw[["date", "open", "high", "low", "close", "volume"]]


def current_price(ticker: str):
    try:
        import requests
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
        })
        t = yf.Ticker(ticker, session=session)
        return round(float(t.fast_info.last_price), 2)
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
        return jsonify({"error": "Internal server error."}), 500


@app.route("/api/backtest", methods=["POST"])
def backtest():
    body     = request.get_json(silent=True) or {}
    ticker   = body.get("ticker",   "").strip().upper()
    strategy = body.get("strategy", "both").strip().lower()
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400
    if strategy not in ("iv", "fusion5", "both"):
        return jsonify({"error": "strategy must be iv, fusion5, or both"}), 400
    try:
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
        return jsonify({"error": "Internal server error."}), 500


@app.route("/api/watchlist", methods=["POST"])
def watchlist():
    body     = request.get_json(silent=True) or {}
    tickers  = body.get("tickers", [])
    strategy = body.get("strategy", "both").strip().lower()
    tickers  = [t.strip().upper() for t in tickers if t.strip()][:20]
    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400

    def generate():
        all_results = []
        for i, ticker in enumerate(tickers):
            try:
                df   = fetch_data(ticker, days=900)
                cp   = current_price(ticker)
                asof = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

                iv_result = run_iv_strategy(df, ticker)
                f5_result = run_fusion5_strategy(df, ticker)
                iv_result["current_price"] = cp
                f5_result["current_price"] = cp
                bt_result = run_backtest(df, ticker, strategy)

                result = {
                    "ticker":   ticker,
                    "price":    cp,
                    "as_of":    asof,
                    "index":    i,
                    "total":    len(tickers),
                    "scan":     {"iv": iv_result, "fusion5": f5_result},
                    "backtest": bt_result,
                }
                all_results.append(result)
                yield f"data: {json.dumps({'type':'ticker','data':result})}\n\n"

            except Exception as e:
                traceback.print_exc()
                yield f"data: {json.dumps({'type':'error','ticker':ticker,'message':str(e)})}\n\n"

        try:
            xlsx_bytes = build_excel(all_results)
            xlsx_b64   = base64.b64encode(xlsx_bytes).decode("utf-8")
            yield f"data: {json.dumps({'type':'done','total':len(all_results),'excel':xlsx_b64})}\n\n"
        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'type':'done','total':len(all_results),'excel':None})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"\n🚀  StockSense AI running on port {port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)   # debug=False — required for Railway
