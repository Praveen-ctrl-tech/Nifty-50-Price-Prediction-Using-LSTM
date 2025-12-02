import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import joblib
import threading

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import tkinter as tk
from tkinter import messagebox

# -------------------------------------------------------
# MODEL CONFIG
# -------------------------------------------------------
TICKER = "^NSEI"
YEARS = 25
WINDOW_SIZE = 60
TEST_DAYS = 365

MODEL_PATH = "nifty_lstm.keras"
SCALER_PATH = "nifty_scaler.pkl"


# -------------------------------------------------------
# FUNCTIONS (MODEL + DATA)
# -------------------------------------------------------
def fetch_data():
    end = datetime.now()
    start = end - timedelta(days=YEARS * 365)

    df = yf.download(TICKER, start=start, end=end, progress=False)
    df = df.dropna()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    return df


def create_sequences(data, win=60):
    X, y = [], []
    for i in range(win, len(data)):
        X.append(data[i-win:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def train_lstm(df, status_label):
    status_label.config(text="Training LSTM model... please wait")

    prices = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X, y = create_sequences(scaled, WINDOW_SIZE)

    split = len(X) - TEST_DAYS
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=32,
        epochs=50,
        verbose=0,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
    )

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    status_label.config(text="Training completed")


def predict_next(df):
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)

    prices = df["Close"].values.reshape(-1, 1)
    scaled = scaler.transform(prices)

    seq = scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
    pred_scaled = model.predict(seq)[0][0]
    pred = scaler.inverse_transform([[pred_scaled]])[0][0]

    return pred


# -------------------------------------------------------
# UI LOGIC
# -------------------------------------------------------
def run_prediction(status_label, result_label):
    try:
        status_label.config(text="Fetching data...")
        df = fetch_data()

        # Train model
        train_lstm(df, status_label)

        # Current price
        current_date = df.index[-1].date()
        current_price = df["Close"].iloc[-1]

        # Prediction
        status_label.config(text="Predicting next day price...")
        predicted_price = predict_next(df)
        predicted_date = current_date + timedelta(days=1)

        # UI OUTPUT
        result_text = (
            f"Current Date: {current_date}\n"
            f"Current Close Price: {current_price:.2f}\n\n"
            f"Predicted Date: {predicted_date}\n"
            f"Predicted Next-Day Close Price: {predicted_price:.2f}"
        )

        result_label.config(text=result_text)
        status_label.config(text="Done")

    except Exception as e:
        messagebox.showerror("Error", str(e))
        status_label.config(text="Error occurred")


def start_prediction_thread(status_label, result_label):
    t = threading.Thread(target=run_prediction, args=(status_label, result_label))
    t.start()


# -------------------------------------------------------
# TKINTER UI
# -------------------------------------------------------
root = tk.Tk()
root.title("NIFTY 50 Predictor (LSTM)")
root.geometry("450x300")

title_label = tk.Label(root, text="NIFTY 50 Price Predictor", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

status_label = tk.Label(root, text="Status: Idle", font=("Arial", 10))
status_label.pack()

predict_button = tk.Button(
    root,
    text="Predict Next-Day Price",
    font=("Arial", 12),
    command=lambda: start_prediction_thread(status_label, result_label)
)
predict_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 11), justify="left")
result_label.pack(pady=10)

root.mainloop()
