# -*- coding: utf-8 -*-
"""
notebook7.py
============
Time Series Forecasting with Deep Learning — Orienta.ai Supporting Analysis
Tecnológico de Monterrey · Advanced AI for Data Science II

Purpose
-------
Evaluates deep learning architectures (CNN, LSTM, CNN-LSTM) for time series
forecasting as part of the predictive analytics work supporting Orienta.ai.

The dataset used here is a retail sales time series (store=1, item=1) to
benchmark model performance before applying similar techniques to career
demand forecasting.

Architecture explored
---------------------
- CNN-1D for local pattern detection in sales windows
- LSTM for sequential dependencies
- Hybrid CNN-LSTM (TimeDistributed)

All models use a sliding window approach:
    Input:  window of 30 consecutive sales values
    Target: sales value `lag` steps ahead (default: 1 day)

Dependencies
------------
    pip install numpy pandas tensorflow scikit-learn matplotlib

Usage
-----
    python notebook7.py

    Expects a CSV file at `train.csv` with columns: date, store, item, sales
    (Kaggle Store Item Demand Forecasting Challenge format)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Conv1D, MaxPooling1D, Flatten,
    LSTM, TimeDistributed
)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from math import sqrt

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 7
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

WINDOW = 29    # Number of past observations used as input features
LAG = 1        # Forecast horizon (steps ahead to predict)
TRAIN_SPLIT = 0.8

# ---------------------------------------------------------------------------
# 1. Data loading and filtering
# ---------------------------------------------------------------------------

def load_data(csv_path: str = "train.csv") -> pd.DataFrame:
    """
    Loads the retail sales dataset and filters to a single store-item pair
    for a clean univariate time series.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Sorted time series with columns: date, store, item, sales.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.query("store == 1 and item == 1").copy()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df)} records | Date range: {df['date'].min()} → {df['date'].max()}")
    return df


# ---------------------------------------------------------------------------
# 2. Windowed supervised dataset construction
# ---------------------------------------------------------------------------

def make_supervised_windows(
    series: np.ndarray,
    window: int,
    lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforms a 1D time series into a supervised learning dataset
    using a sliding window approach.

    For each position i from `window` to `len(series) - lag`:
        X[i] = series[i-window : i+1]   (window+1 values)
        y[i] = series[i + lag]           (target value lag steps ahead)

    Parameters
    ----------
    series : np.ndarray
        1D array of time series values.
    window : int
        Number of past time steps to include in each input sample.
    lag : int
        Number of steps ahead to predict.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X of shape (n_samples, window+1) and y of shape (n_samples,).
    """
    X, y = [], []
    for i in range(window, len(series) - lag):
        X.append(series[i - window : i + 1])
        y.append(series[i + lag])
    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------

def preprocess(
    X: np.ndarray,
    y: np.ndarray,
    train_split: float = TRAIN_SPLIT
) -> dict:
    """
    Splits data into train/test sets and applies StandardScaler normalization.
    Scalers are fit only on training data to prevent data leakage.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, window+1).
    y : np.ndarray
        Target vector of shape (n_samples,).
    train_split : float
        Fraction of data to use for training.

    Returns
    -------
    dict
        Dictionary containing scaled train/test arrays and fitted scalers.
    """
    cut = int(len(X) * train_split)

    X_train, X_test = X[:cut], X[cut:]
    y_train, y_test = y[:cut], y[cut:]

    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

    return {
        "X_train": scaler_x.transform(X_train),
        "X_test":  scaler_x.transform(X_test),
        "y_train": scaler_y.transform(y_train.reshape(-1, 1)).ravel(),
        "y_test":  scaler_y.transform(y_test.reshape(-1, 1)).ravel(),
        "scaler_y": scaler_y,
        "n_train":  cut,
    }


def reshape_3d(X: np.ndarray) -> np.ndarray:
    """Reshapes 2D feature matrix to 3D (samples, timesteps, features) for CNN/LSTM."""
    return X.reshape((X.shape[0], X.shape[1], 1))


# ---------------------------------------------------------------------------
# 4. Model definitions
# ---------------------------------------------------------------------------

def build_cnn(input_shape: tuple) -> Sequential:
    """
    1D Convolutional Neural Network for time series forecasting.
    Captures local temporal patterns via convolutional filters.

    Architecture: Conv1D → MaxPool → Flatten → Dense → Dense
    """
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation="relu"),
        Dense(1),
    ], name="CNN_Forecaster")
    model.compile(optimizer="adam", loss="mse")
    return model


def build_lstm(input_shape: tuple) -> Sequential:
    """
    LSTM network for capturing long-range sequential dependencies.

    Architecture: LSTM(50) → Dense
    """
    model = Sequential([
        LSTM(50, activation="relu", input_shape=input_shape),
        Dense(1),
    ], name="LSTM_Forecaster")
    model.compile(optimizer="adam", loss="mse")
    return model


def build_cnn_lstm(input_shape: tuple, subsequences: int = 2) -> Sequential:
    """
    Hybrid CNN-LSTM model using TimeDistributed layers.
    CNN extracts local features; LSTM learns temporal relationships across subsequences.

    Parameters
    ----------
    input_shape : tuple
        Shape per subsequence: (timesteps_per_subseq, features).
    subsequences : int
        Number of subsequences to split the input window into.
    """
    model = Sequential([
        TimeDistributed(Conv1D(filters=64, kernel_size=1, activation="relu"),
                        input_shape=(subsequences,) + input_shape),
        TimeDistributed(MaxPooling1D(pool_size=1)),
        TimeDistributed(Flatten()),
        LSTM(50, activation="relu"),
        Dense(1),
    ], name="CNN_LSTM_Forecaster")
    model.compile(optimizer="adam", loss="mse")
    return model


# ---------------------------------------------------------------------------
# 5. Training and evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y: StandardScaler,
    epochs: int = 20,
    batch_size: int = 32,
) -> dict:
    """
    Trains a model and evaluates it on the test set.
    Returns RMSE on original (inverse-scaled) units.

    Parameters
    ----------
    model : Sequential
        Compiled Keras model.
    X_train, y_train : np.ndarray
        Scaled training data.
    X_test, y_test : np.ndarray
        Scaled test data.
    scaler_y : StandardScaler
        Fitted scaler used to inverse-transform predictions.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.

    Returns
    -------
    dict
        Dictionary with 'rmse', 'history', and 'predictions'.
    """
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=0,
    )

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    rmse = sqrt(mean_squared_error(y_true, y_pred))
    print(f"  {model.name} → RMSE: {rmse:.4f}")

    return {
        "rmse": rmse,
        "history": history,
        "predictions": y_pred,
        "actuals": y_true,
    }


# ---------------------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------------------

def plot_results(results: dict, n_plot: int = 100) -> None:
    """
    Plots actual vs predicted values for each model and their training loss curves.

    Parameters
    ----------
    results : dict
        Dictionary mapping model name to its evaluation results dict.
    n_plot : int
        Number of test points to display in the forecast plot.
    """
    fig, axes = plt.subplots(len(results), 2, figsize=(14, 4 * len(results)))

    for idx, (name, res) in enumerate(results.items()):
        # Forecast comparison
        ax1 = axes[idx, 0]
        ax1.plot(res["actuals"][:n_plot], label="Actual", linewidth=1.5)
        ax1.plot(res["predictions"][:n_plot], label="Predicted", linewidth=1.5, linestyle="--")
        ax1.set_title(f"{name} — Forecast (RMSE: {res['rmse']:.4f})")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Loss curve
        ax2 = axes[idx, 1]
        ax2.plot(res["history"].history["loss"], label="Train Loss")
        ax2.plot(res["history"].history["val_loss"], label="Val Loss")
        ax2.set_title(f"{name} — Training Loss")
        ax2.legend()
        ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("forecast_results.png", dpi=150)
    plt.show()
    print("Plot saved to forecast_results.png")


# ---------------------------------------------------------------------------
# 7. Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Time Series Forecasting — Orienta.ai")
    print("=" * 60)

    # Load data
    df = load_data("train.csv")
    sales = df["sales"].values

    # Build supervised dataset
    X, y = make_supervised_windows(sales, window=WINDOW, lag=LAG)
    print(f"\nDataset shape — X: {X.shape}, y: {y.shape}")

    # Preprocess
    data = preprocess(X, y)
    X_train_3d = reshape_3d(data["X_train"])
    X_test_3d  = reshape_3d(data["X_test"])

    input_shape = (X_train_3d.shape[1], 1)

    # Train and evaluate all models
    print("\nTraining models...")
    results = {}

    results["CNN"] = train_and_evaluate(
        build_cnn(input_shape),
        X_train_3d, data["y_train"],
        X_test_3d,  data["y_test"],
        data["scaler_y"],
    )

    results["LSTM"] = train_and_evaluate(
        build_lstm(input_shape),
        X_train_3d, data["y_train"],
        X_test_3d,  data["y_test"],
        data["scaler_y"],
    )

    # CNN-LSTM requires reshaping into subsequences
    subsequences = 2
    timesteps = X_train_3d.shape[1] // subsequences
    X_train_cnn_lstm = X_train_3d.reshape(-1, subsequences, timesteps, 1)
    X_test_cnn_lstm  = X_test_3d.reshape(-1, subsequences, timesteps, 1)

    results["CNN-LSTM"] = train_and_evaluate(
        build_cnn_lstm((timesteps, 1), subsequences),
        X_train_cnn_lstm, data["y_train"],
        X_test_cnn_lstm,  data["y_test"],
        data["scaler_y"],
    )

    # Summary
    print("\n" + "=" * 40)
    print("RESULTS SUMMARY")
    print("=" * 40)
    for name, res in results.items():
        print(f"  {name:<12} RMSE = {res['rmse']:.4f}")

    best = min(results, key=lambda k: results[k]["rmse"])
    print(f"\n  Best model: {best} (RMSE = {results[best]['rmse']:.4f})")

    # Visualize
    plot_results(results)


if __name__ == "__main__":
    main()
