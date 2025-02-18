import pandas as pd
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit

def fetch_data(stock_symbol: str):
    try:
        data = yf.download(stock_symbol.upper().strip(), period = '10y', interval = '1d' )
    except Exception as e:
        print(f'An error occurred while downloading data: {e}')
        return None

    if data is None or data.empty:
        print(f'No data found for {stock_symbol}')
        return None

    data['Normalized'] = (data['Close'] - data['Close'].mean()) / data['Close'].std()
    return data

def prepare_data(data: pd.DataFrame, window: int):
    prices = data['Normalized'].to_numpy()
    x = []
    y = []
    for i in range(len(prices) - window):
        x.append(prices[i:(i + window)])
        y.append(prices[i + window])
    return x, y

def split_data(x, y, scale):
    split_index = int(len(x) * scale)
    x_train = x[:split_index]
    x_val = x[split_index:]
    y_train = y[:split_index]
    y_val = y[split_index:]
    return x_train, x_val, y_train, y_val

def split_data_walk(x,y,n_splits=5):
    DataSplit = TimeSeriesSplit(n_splits=n_splits)
    x_train = []
    x_val = []
    y_train = []
    y_val = []
    for train_index, eval_index in DataSplit.split(x):
        x_train.append(x[train_index])
        x_val.append(x[eval_index])
        y_train.append(y[train_index])
        y_val.append(y[eval_index])
    return x_train, x_val, y_train, y_val




