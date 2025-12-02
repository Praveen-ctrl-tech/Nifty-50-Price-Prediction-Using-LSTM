**📈 NIFTY-50 Next-Day Price Prediction (LSTM + UI)**

This project implements a Next-Day Stock Price Prediction System for NIFTY-50 using a Long Short-Term Memory (LSTM) deep-learning model.
The system automatically fetches historical NIFTY-50 market data from Yahoo Finance, trains an LSTM model, predicts the next day’s closing price, and displays the results in a simple user interface (UI).

The goal of this project is to help users understand short-term market trends and experiment with time-series forecasting using modern deep-learning techniques.

**📌 Features
**1. Automatic Data Fetching****

Retrieves 25 years of official NIFTY-50 (India) historical data

Uses Yahoo Finance API via yfinance

Includes OHLCV (Open, High, Low, Close, Volume) data

**2. Deep Learning Model – LSTM**

Uses a multi-layer LSTM network with dropout

Trained on normalized closing prices

Builds automated sliding-window sequences (60-day lookback)

Produces a prediction for the next trading day

**3. User Interface (UI)**

Built using Tkinter, simple and lightweight

User clicks “Predict Next-Day Price”

Model is trained and prediction is displayed

Shows:

Current Date

Current Closing Price

Predicted Next Date

Predicted Next-Day Closing Price

**4. Background Non-Blocking Execution**

Uses Python threading to ensure the UI does not freeze

Good for slow CPUs or long training cycles

🧠 System Architecture

Yahoo Finance (API)

        ↓
Data Preprocessing
    - Clean missing values
    - Normalize data (MinMax)
    - Create 60-day sliding window
    
        ↓
LSTM Model
    - 64-unit LSTM layer
    - 32-unit LSTM layer
    - Dense output layer
    
        ↓
Next-Day Price Prediction

        ↓
Desktop UI (Tkinter)
    - Predict button
    - Runs model
    - Shows current price & predicted price


🛠 Technologies Used
Languages

Python 3.12+ / 3.13

Libraries & Frameworks

| Category         | Library                |
| ---------------- | ---------------------- |
| Data Fetching    | `yfinance`             |
| Data Processing  | `pandas`, `numpy`      |
| ML Preprocessing | `MinMaxScaler`         |
| Deep Learning    | `TensorFlow` / `Keras` |
| UI Framework     | `Tkinter`              |
| Storage          | `joblib`               |

🎯 Prediction Output Example

After clicking the Predict button, the UI displays:

Current Date: 2025-12-02
Current Close Price: 24680.55

Predicted Date: 2025-12-03
Predicted Next-Day Close Price: 24752.81

▶️ How to Run
1. Install Dependencies
   pip install yfinance pandas numpy tensorflow joblib scikit-learn
2. Run the Application
   python nifty_50_predictor.py
3. Click “Predict Next-Day Price” in UI

