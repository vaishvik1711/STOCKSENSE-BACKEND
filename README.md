# StockSense AI — IV Strategy
## Setup Instructions

### 1. Install dependencies
```bash
pip install flask flask-cors yfinance pandas numpy
```

### 2. Project structure
```
your-folder/
├── app.py           ← Flask backend
├── iv_strategy.py   ← IV Strategy engine
├── stocksense.html  ← Frontend UI
└── requirements.txt
```

### 3. Run the backend
```bash
python app.py
```
You'll see: 🚀 StockSense AI Backend running at http://localhost:5000

### 4. Open the frontend
Open `stocksense.html` directly in your browser (double-click it).
No web server needed for the frontend.

---

## How the IV Strategy works

| Step | Rule |
|------|------|
| Consolidation | ≥15 trading days where (Highest High − Lowest Low) / Lowest Low ≤ 10% |
| IV Candle | Green candle (close > open), volume > 30-day average AND largest in 30 days |
| Entry | First daily close above IV candle high within 15 trading days → enter at market close |
| Stop Loss | max(IV candle low, entry × 0.85) — whichever is a higher price |
| Target | Entry price × 1.30 (30% gain) |

## API

**POST** `http://localhost:5000/api/analyze`
```json
{ "ticker": "AAPL" }
```

Response statuses:
- `ENTRY_TRIGGERED` — trade signal active with full plan
- `WATCHING` — IV candle found, waiting for close above IV high
- `EXPIRED` — 15-day window passed with no trigger
- `NO_SIGNAL` — no valid IV setup found in last 60 days
