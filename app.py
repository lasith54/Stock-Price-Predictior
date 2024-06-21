import numpy as np
import yfinance as yf
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from copy import deepcopy as dc

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Load the model
device = torch.device('cpu')
model = LSTM(3, 64, 2)
model.load_state_dict(torch.load('lstm_model.pth', map_location=device))
model.eval()

st.header('Stock Price Predictor')

ticker = st.text_input('Enter Stock Ticker', 'TSLA')

ticker_info = yf.Ticker(ticker)
stock_info = ticker_info.history(period='max')
start = "2017-06-21"
end = datetime.datetime.now().strftime('%Y-%m-%d')

data = yf.download(ticker, start, end)

st.subheader('Historical Stock Prices')
st.write(data)

data_train = data.iloc[:int(len(data)*0.80)]
data_test = data.iloc[int(len(data)*0.80):]

scaler = StandardScaler()

features = ['Open', 'High', 'Low']
target = ['Close']

data_train_scaled = scaler.fit_transform(data_train[features + target])
data_test_scaled = scaler.transform(data_test[features + target])

past_100_days = data_train_scaled[-100:]
data_test_scaled = np.concatenate((past_100_days, data_test_scaled), axis=0)

st.subheader('MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(10,8))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.legend()
plt.show()
st.pyplot(fig1)

st.subheader('MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(10,8))
plt.plot(ma_100_days, 'r')
plt.plot(data.Close, 'g')
plt.legend()
plt.show()
st.pyplot(fig2)

n_past = 30
n_future = 1

X = []
y = []

for i in range(n_past, len(data_test_scaled) - n_future + 1):
    X.append(data_test_scaled[i - n_past:i, :-1])
    y.append(data_test_scaled[i + n_future - 1, -1])

X,y = np.array(X), np.array(y)

X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)

# Make predictions
with torch.no_grad():
    model = model.eval()
    predictions = model(X).squeeze().numpy()

predictions = predictions.reshape(-1, 1)
dummy_features = np.zeros((predictions.shape[0], 3))
predictions_with_dummy = np.concatenate((predictions, dummy_features), axis=1)
predictions_original_scale = scaler.inverse_transform(predictions_with_dummy)

test_predictions = predictions_original_scale[:, 0]

y_original_scale = scaler.inverse_transform(np.concatenate((y.reshape(-1, 1), dummy_features), axis=1))[:, 0]

st.subheader('Original Price vs Predicted Price')
fig3 = plt.figure(figsize=(10,8))
plt.plot(test_predictions, 'r', label='Original Price')
plt.plot(y_original_scale, 'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig3)