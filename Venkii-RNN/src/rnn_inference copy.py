"""
Inference Script for SimpleRNN Traffic Model
=============================================
Loads saved model + scaler, accepts last N rows from dataset
(sequence length read from saved scaler), and predicts the
next timestep traffic volume.
"""

import os
import pickle

import numpy as np
import pandas as pd
from keras.models import load_model

# Suppress TensorFlow info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rnn_traffic_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "rnn_scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "raw-data", "feature_engineered_dataset.csv")
DEFAULT_SEQUENCE_LENGTH = 48


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_artifacts(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """Load saved model and scaler artefacts."""
    model = load_model(model_path, compile=False)
    with open(scaler_path, "rb") as f:
        scaler_data = pickle.load(f)
    return model, scaler_data


def get_sequence_length(scaler_data):
    """Read sequence length from saved scaler or fall back to default."""
    return scaler_data.get("sequence_length", DEFAULT_SEQUENCE_LENGTH)


def detect_datetime_col(df):
    """Auto-detect datetime column."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        if df[col].dtype == "object":
            try:
                pd.to_datetime(df[col].head(100))
                return col
            except (ValueError, TypeError):
                continue
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                pd.to_datetime(df[col])
                return col
            except (ValueError, TypeError):
                continue
    return None


def prepare_input(df, scaler_data, n_rows):
    """Prepare the last n_rows as a model-ready input tensor."""
    target_col = scaler_data["target_col"]
    feature_scaler = scaler_data["feature_scaler"]

    datetime_col = detect_datetime_col(df)
    if datetime_col:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(datetime_col).reset_index(drop=True)

    df_last = df.tail(n_rows).copy()
    df_numeric = df_last.select_dtypes(include=[np.number])
    X = df_numeric.drop(columns=[target_col], errors="ignore").values

    X_scaled = feature_scaler.transform(X)
    return X_scaled.reshape(1, n_rows, -1)


def predict_next(model, X, scaler_data):
    """Run prediction and inverse-transform to original scale."""
    target_scaler = scaler_data["target_scaler"]
    pred_scaled = model.predict(X, verbose=0).flatten()
    pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    return pred[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 50)
    print("  SimpleRNN — Traffic Volume Inference")
    print("=" * 50)

    model, scaler_data = load_artifacts()
    seq_len = get_sequence_length(scaler_data)
    print(f"Model loaded : {MODEL_PATH}")
    print(f"Scaler loaded: {SCALER_PATH}")
    print(f"Sequence len : {seq_len}")

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset rows : {len(df)}")

    X = prepare_input(df, scaler_data, seq_len)
    print(f"Input shape  : {X.shape}")

    prediction = predict_next(model, X, scaler_data)
    print(f"\n>>> Predicted next traffic volume: {prediction:.2f}")


if __name__ == "__main__":
    main()
