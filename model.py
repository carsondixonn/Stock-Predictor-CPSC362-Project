import pandas as pd
from sklearn.linear_model import LinearRegression
from yahoofinancials import YahooFinancials
stock = input("What stock: ")

highs = []
lows = []
opens = []
closes = []

yahoo_financials = YahooFinancials(str(stock))
stats=(yahoo_financials.get_historical_price_data("2016-01-01", "2026-08-26", "daily"))

i = 0
for date in stats[str(stock)]["prices"]:
    if i == 0:
        i += 1
        continue
    highs.append(date["high"])
    lows.append(date["low"])
    opens.append(date["open"])
    closes.append(date["close"])
    i += 1
print("No, of data pts", i)

total = []
totalopens = []
for j in range(4):
    opens.append(0)

for i in range(i-1):
    total.append([opens[i], lows[i], highs[i], closes[i]])


def Predictor(lst, last_row):
    training = lst[0:i-last_row]
    validations = lst[i-last_row:]

    df = pd.DataFrame(training, dtype=float)
    XTrain = df.iloc[:, :-1]
    yTrain = df.iloc[:, [-1]]

    model = LinearRegression()
    model.fit(XTrain, yTrain)

    xvalidations = [row[:-1] for row in validations]
    yvalidations = [row[-1] for row in validations]

    prediction = model.predict(xvalidations)

    current_price = yvalidations[-1]
    predicted_price = prediction[-1][0]

    print("Current price:", current_price)
    print("Predicted price:", predicted_price)

Predictor(total, 1)




