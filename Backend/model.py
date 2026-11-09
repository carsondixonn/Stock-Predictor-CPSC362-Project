import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

#for database
from database import create_db_and_tables
#for auth
from auth import router as auth_router
#for portfolio
from portfolio import router as portfolio_router

#uvicorn model:app --reload
app = FastAPI()

#for auth
app.include_router(auth_router)
#for portfolio
app.include_router(portfolio_router)

#for database
create_db_and_tables()

BASE_DIR = Path(__file__).parent

@app.get("/login.html", include_in_schema=False)
def login_page():
    return FileResponse(BASE_DIR / "login.html")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(BASE_DIR / "index.html")

@app.get("/index.html", include_in_schema=False)
def home_page():
    return FileResponse(BASE_DIR / "index.html")

@app.get("/model.html", include_in_schema=False)
def model_page():
    return FileResponse(BASE_DIR / "model.html")

@app.get("/index.css")
def css():
    return FileResponse(BASE_DIR / "index.css", media_type="text/css")

@app.get("/auth.js", include_in_schema=False)
def auth_js():
    return FileResponse(BASE_DIR / "auth.js", media_type="text/javascript")

@app.get("/ticker-search.js", include_in_schema=False)
def ticker_search_js():
    return FileResponse(BASE_DIR / "ticker-search.js", media_type="text/javascript")

@app.get("/portfolio.html", include_in_schema=False)
def portfolio_page():
    return FileResponse(BASE_DIR / "portfolio.html")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

VALID_INTERVALS = ("daily", "weekly", "monthly", "yearly")

# "ME"/"YE" only exist in pandas 2.2+; older versions use "M"/"Y".
_PANDAS_VERSION = tuple(int(p) for p in pd.__version__.split(".")[:2])
_NEW_ALIASES = _PANDAS_VERSION >= (2, 2)
RESAMPLE_RULES = {
    "weekly": "W",
    "monthly": "ME" if _NEW_ALIASES else "M",
    "yearly": "YE" if _NEW_ALIASES else "Y",
}

# How much history to train on when the caller doesn't give a start date.
# Recent data only: decades of old prices describe a very different market.
# (None = everything available; yearly bars are too scarce to trim.)
LOOKBACK_YEARS = {"daily": 3, "weekly": 5, "monthly": 10, "yearly": None}

# Minimum number of (features -> future return) rows needed to fit the model.
# Yearly data can't realistically give 30 rows, so it gets a lower floor.
MIN_TRAIN_ROWS = {"daily": 30, "weekly": 30, "monthly": 30, "yearly": 15}

# The held-out test needs at least this many training rows left over.
MIN_EVAL_TRAIN_ROWS = 15

HISTORY_POINTS = 30          # closing prices returned for the portfolio charts
MARKET_TZ = "America/New_York"
MARKET_CLOSE_HOUR = 16       # a daily bar is "final" after 4pm Eastern


class PredictionError(Exception):
    """A problem with the request or the data (returned as HTTP 400)."""


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def parse_horizon(raw):
    text = str(raw).strip() if raw is not None else ""
    if text == "":
        raise PredictionError("Horizon is required (a whole number of at least 1).")
    try:
        horizon = int(text)
    except ValueError:
        raise PredictionError(f"Horizon must be a whole number, got '{text}'.")
    if horizon < 1:
        raise PredictionError("Horizon must be at least 1.")
    return horizon


def parse_date(raw, name):
    if raw is None or str(raw).strip() == "":
        return None
    try:
        ts = pd.to_datetime(str(raw).strip())
    except (ValueError, TypeError):
        raise PredictionError(f"Invalid {name} date '{raw}'. Use a format like 2024-01-31.")
    if pd.isna(ts):
        raise PredictionError(f"Invalid {name} date '{raw}'. Use a format like 2024-01-31.")
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch_data(ticker, interval, start=None, end=None):
    """Daily OHLC for the ticker, trimmed to the training window."""
    years = LOOKBACK_YEARS[interval]
    if start is None and years is not None:
        start = pd.Timestamp.now().normalize() - pd.DateOffset(years=years)

    kwargs = {"progress": False, "auto_adjust": True}
    if start is not None:
        kwargs["start"] = start.strftime("%Y-%m-%d")
    else:
        kwargs["period"] = "max"
    if end is not None:
        kwargs["end"] = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # end is exclusive

    df = yf.download(ticker, **kwargs)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index <= end]
    return df


def resample_data(df, interval):
    if interval == "daily":
        return df
    return df.resample(RESAMPLE_RULES[interval]).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }).dropna()


def drop_incomplete_bar(df, end=None, now=None):
    """Remove the last bar if its period hasn't finished.

    Bars are labelled with the last calendar day of their period (the day itself
    for daily, Sunday for weekly, month-end, year-end). A bar is final once that
    day is over (or, for today, once the market has closed). If the user gave an
    end date, any period reaching past it is cut short too, so it goes as well.
    """
    now = now if now is not None else pd.Timestamp.now(tz=MARKET_TZ)
    today = now.tz_localize(None).normalize()
    cutoff = today if now.hour >= MARKET_CLOSE_HOUR else today - pd.Timedelta(days=1)
    if end is not None:
        cutoff = min(cutoff, end)
    return df[df.index <= cutoff]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_features(df):
    """Scale-free features (no raw prices, so $10 and $1000 stocks look alike)."""
    close = df["Close"]
    ret1 = close.pct_change()
    return pd.DataFrame({
        "ret_1": ret1,                              # latest period's return
        "ret_1_prev": ret1.shift(1),                # the one before it
        "ret_5": close.pct_change(5),               # short-term momentum
        "intra": close / df["Open"] - 1,            # close vs open
        "range": (df["High"] - df["Low"]) / close,  # how wide the bar was
        "vol_5": ret1.rolling(5).std(),             # recent volatility
    }, index=df.index)


def new_model():
    # Ridge keeps coefficients small, which matters because returns are mostly noise.
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def evaluate(X, y, closes_at_rows, horizon):
    """Train on the older rows, test on the most recent ones.

    A gap of `horizon` rows sits between the two sets so that no training target
    looks at prices the test rows also use. Compares the model with the naive
    "the price stays the same" guess.
    """
    n = len(X)
    test_size = max(5, n // 5)
    train_end = n - test_size - horizon
    if train_end < MIN_EVAL_TRAIN_ROWS:
        return None

    model = new_model().fit(X.iloc[:train_end], y.iloc[:train_end])
    pred_ret = model.predict(X.iloc[n - test_size:])

    base = closes_at_rows.iloc[n - test_size:].to_numpy()
    actual = base * (1 + y.iloc[n - test_size:].to_numpy())
    model_price = base * (1 + pred_ret)

    model_mae = float((abs(model_price - actual) / actual).mean() * 100)
    naive_mae = float((abs(base - actual) / actual).mean() * 100)
    beats = model_mae < naive_mae

    if beats:
        note = (f"On the last {test_size} periods the model's average error was "
                f"{model_mae:.2f}% vs {naive_mae:.2f}% for guessing the price stays the same.")
    else:
        note = (f"On the last {test_size} periods the model's average error was "
                f"{model_mae:.2f}%, no better than the {naive_mae:.2f}% from simply guessing "
                f"the price stays the same. Treat this prediction with caution.")

    return {
        "test_points": test_size,
        "model_mae_pct": round(model_mae, 3),
        "naive_mae_pct": round(naive_mae, 3),
        "beats_naive": beats,
        "note": note,
    }


def target_period(last_label, interval, horizon):
    """The date the prediction is for, and what kind of period that is."""
    if interval == "daily":
        return last_label + pd.offsets.BDay(horizon), "day"  # weekends skipped, holidays not
    if interval == "weekly":
        return last_label + pd.Timedelta(weeks=horizon), "week ending"
    if interval == "monthly":
        return last_label + pd.offsets.MonthEnd(horizon), "month ending"
    return last_label + pd.offsets.YearEnd(horizon), "year ending"


def predict_stock(ticker, interval, horizon, start=None, end=None):
    # --- validate everything before touching the network ---
    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise PredictionError("Please enter a ticker symbol.")

    interval = (interval or "").strip().lower()
    if interval not in VALID_INTERVALS:
        raise PredictionError(
            f"Interval must be one of: {', '.join(VALID_INTERVALS)} (got '{interval}')."
        )

    horizon = parse_horizon(horizon)
    start = parse_date(start, "start")
    end = parse_date(end, "end")
    if start is not None and end is not None and start > end:
        raise PredictionError("Start date must be on or before the end date.")

    # --- data ---
    df = fetch_data(ticker, interval, start, end)
    if df.empty:
        raise PredictionError(f"No data found for ticker '{ticker}' in that date range.")

    df = resample_data(df, interval)
    df = drop_incomplete_bar(df, end)

    # --- features and targets (all as returns) ---
    features = build_features(df)
    target = df["Close"].shift(-horizon) / df["Close"] - 1  # return over the next `horizon` bars

    train_mask = features.notna().all(axis=1) & target.notna()
    X, y = features[train_mask], target[train_mask]
    closes_at_rows = df["Close"][train_mask]

    need = MIN_TRAIN_ROWS[interval]
    if len(X) < need:
        raise PredictionError(
            f"Not enough data: {len(X)} usable {interval} training rows for horizon={horizon}, "
            f"need at least {need}. Try an earlier start date, a shorter horizon, "
            f"or a shorter interval."
        )

    latest = features.iloc[[-1]]
    if latest.isna().any(axis=None):
        raise PredictionError("Not enough recent data to compute the model inputs.")

    evaluation = evaluate(X, y, closes_at_rows, horizon)

    # Final model: fit on every known pair, predict from the newest bar.
    model = new_model().fit(X, y)
    predicted_return = float(model.predict(latest)[0])

    current_price = float(df["Close"].iloc[-1])
    last_label = df.index[-1]
    target_date, target_kind = target_period(last_label, interval, horizon)

    return {
        "ticker": ticker,
        "interval": interval,
        "horizon": horizon,
        "current_price": current_price,
        "predicted_price": current_price * (1 + predicted_return),
        "predicted_return_pct": predicted_return * 100,
        "last_data_date": last_label.strftime("%Y-%m-%d"),
        "target_date": target_date.strftime("%Y-%m-%d"),
        "target_period": target_kind,
        "data_points_used": len(df),
        "training_rows": len(X),
        "evaluation": evaluation,  # None if there wasn't enough data to hold some back
        "history": [float(v) for v in df["Close"].iloc[-HISTORY_POINTS:]],
    }


@app.get("/predict")
def predict(ticker: str, interval: str = "daily", horizon: str = "1", start: str = None, end: str = None):
    # horizon is a string on purpose: an empty form field would otherwise be a 422
    # with a list-shaped body that the page shows as [object Object].
    try:
        return predict_stock(ticker, interval, horizon, start, end)
    except PredictionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Live quotes (for the portfolio page to poll)
# ---------------------------------------------------------------------------

QUOTE_TTL_OPEN = 20        # seconds to reuse a price while the market is open
QUOTE_TTL_CLOSED = 300     # prices barely change when closed, so ask Yahoo less
MAX_QUOTE_TICKERS = 25
_quote_cache = {}          # ticker -> (fetched_at, price)


def market_is_open(now=None):
    """Regular US hours, Mon-Fri 9:30-16:00 Eastern (market holidays not handled)."""
    now = now if now is not None else pd.Timestamp.now(tz=MARKET_TZ)
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= minutes < MARKET_CLOSE_HOUR * 60


def latest_price(ticker, ttl):
    """Most recent price for one ticker (Yahoo is ~15 min delayed), or None."""
    hit = _quote_cache.get(ticker)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    try:
        price = yf.Ticker(ticker).fast_info.last_price
        price = float(price)
    except Exception:
        return hit[1] if hit else None   # fall back to the last good price
    if price != price or price <= 0:     # NaN or nonsense
        return hit[1] if hit else None
    _quote_cache[ticker] = (time.time(), price)
    return price


@app.get("/quote")
def quote(tickers: str):
    """Latest prices for several tickers at once: /quote?tickers=AAPL,MSFT"""
    names = list(dict.fromkeys(t.strip().upper() for t in tickers.split(",") if t.strip()))
    if not names:
        raise HTTPException(status_code=400, detail="Give at least one ticker.")
    if len(names) > MAX_QUOTE_TICKERS:
        raise HTTPException(status_code=400, detail=f"At most {MAX_QUOTE_TICKERS} tickers per request.")

    is_open = market_is_open()
    ttl = QUOTE_TTL_OPEN if is_open else QUOTE_TTL_CLOSED
    with ThreadPoolExecutor(max_workers=8) as pool:
        prices = list(pool.map(lambda t: latest_price(t, ttl), names))

    return {
        "market_open": is_open,
        "as_of": pd.Timestamp.now(tz=MARKET_TZ).isoformat(),
        "quotes": {t: p for t, p in zip(names, prices) if p is not None},
    }


# ---------------------------------------------------------------------------
# Ticker search (autocomplete on the prediction and portfolio pages)
# ---------------------------------------------------------------------------

SEARCH_TTL = 600           # seconds to remember a search
MAX_SEARCH_RESULTS = 8
_search_cache = {}         # lowercase query -> (fetched_at, results)


@app.get("/search")
def search_tickers(q: str):
    """Find tickers by company name or symbol: /search?q=apple"""
    q = q.strip()
    if not q or len(q) > 40:
        return {"results": []}

    key = q.lower()
    hit = _search_cache.get(key)
    if hit and time.time() - hit[0] < SEARCH_TTL:
        return {"results": hit[1]}

    try:
        found = yf.Search(q, max_results=MAX_SEARCH_RESULTS, news_count=0).quotes
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ticker search unavailable: {e}")

    results = []
    for item in found:
        if item.get("quoteType") not in ("EQUITY", "ETF") or not item.get("symbol"):
            continue
        results.append({
            "symbol": item["symbol"],
            "name": item.get("shortname") or item.get("longname") or item["symbol"],
            "exchange": item.get("exchDisp") or "",
        })
    _search_cache[key] = (time.time(), results)
    return {"results": results}
