import pandas as pd
from sklearn.linear_model import LinearRegression
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
#uvicorn main:app --reload
app = FastAPI()

# Allow the React dev server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def fetch_data(ticker, start=None, end=None):
    df = yf.download(ticker, period="max", progress=False)
    df = df.dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]
    return df


def resample_data(df, interval):
    if interval == "daily":
        return df
    rule = {"weekly": "W", "monthly": "M", "yearly": "A"}[interval]
    return df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }).dropna()


def predict_stock(ticker, interval, horizon=1, start=None, end=None):
    df = fetch_data(ticker, start, end)
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'")

    df = resample_data(df, interval)

    opens = df["Open"].tolist()
    lows = df["Low"].tolist()
    highs = df["High"].tolist()
    closes = df["Close"].tolist()

    total = [[opens[i], lows[i], highs[i], closes[i]] for i in range(len(closes))]

    if len(total) <= horizon + 1:
        raise ValueError(
            f"Not enough data points ({len(total)}) for interval='{interval}' "
            f"with horizon={horizon}."
        )

    X = total[:-horizon]
    y = [row[3] for row in total[horizon:]]

    XTrain = pd.DataFrame(X[:-1], dtype=float)
    yTrain = pd.DataFrame(y[:-1], dtype=float)
    XVal = [X[-1]]

    model = LinearRegression()
    model.fit(XTrain, yTrain)
    prediction = model.predict(XVal)

    return {
        "ticker": ticker,
        "interval": interval,
        "horizon": horizon,
        "current_price": closes[-1],
        "predicted_price": float(prediction[0][0]),
        "data_points_used": len(total),
    }


@app.get("/predict")
def predict(ticker: str, interval: str = "daily", horizon: int = 1, start: str = None, end: str = None):
    try:
        return predict_stock(ticker.upper(), interval, horizon, start, end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
