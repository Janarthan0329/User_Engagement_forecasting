# backend/forecast/model_loader.py
import importlib.util
import joblib
import numpy as np
import os
from pathlib import Path
from typing import Any, Dict

# Try to use Django's BASE_DIR if available (safer in deployment),
# otherwise fall back to resolving from this file
try:
    from django.conf import settings
    DJANGO_BASE_DIR = Path(settings.BASE_DIR)
except Exception:
    DJANGO_BASE_DIR = Path(__file__).resolve().parent.parent

# Prefer tensorflow.keras if available, otherwise fall back to standalone keras
try:
    tf_models = importlib.import_module("tensorflow.keras.models")
    load_model = tf_models.load_model
except ImportError:
    try:
        keras_models = importlib.import_module("keras.models")
        load_model = keras_models.load_model
    except ImportError:
        raise ImportError(
            "Neither tensorflow.keras.models nor keras.models is available; "
            "please install tensorflow or keras."
        )

# -------------------------------------------------------------------
# Model root directory (can be overridden with env var MODEL_ROOT)
# -------------------------------------------------------------------
MODEL_ROOT_ENV = os.environ.get("MODEL_ROOT")
if MODEL_ROOT_ENV:
    MODELS_DIR = Path(MODEL_ROOT_ENV).resolve()
else:
    # default: backend/models relative to project root
    MODELS_DIR = DJANGO_BASE_DIR / "models"

# ---------------- V3 PATHS: Prophet + LSTM (win=28) ----------------
V3_DIR = MODELS_DIR / "v3"
V3_LOOK_BACK = 28

V3_PATHS = {
    "lstm": V3_DIR / "lstm_best_win28.keras",
    "x_scaler": V3_DIR / "x_scaler.npy",
    "y_scaler": V3_DIR / "y_scaler.npy",
}

# ---------------- V4 PATHS: SARIMAX + GRU (win=28) -----------------
V4_DIR = MODELS_DIR / "v4"
V4_LOOK_BACK = 28

V4_PATHS = {
    "sarimax": V4_DIR / "sarimax_v4_model.pkl",
    "gru": V4_DIR / "gru_v4_deep28_win28.keras",
    "x_scaler": V4_DIR / "x_scaler_v4.npy",
    "y_scaler": V4_DIR / "y_scaler_v4.npy",
}

# ------------------------ Simple cache -----------------------------
_cache: Dict[str, Any] = {
    "v3_lstm": None,
    "v3_x_scaler": None,
    "v3_y_scaler": None,
    "v4_sarimax": None,
    "v4_gru": None,
    "v4_x_scaler": None,
    "v4_y_scaler": None,
}


def _ensure_exists(path: Path, label: str) -> None:
    """
    Raise a clear error if a required artifact file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Required {label} file not found at: {path}. "
            f"Check that model artifacts are deployed under '{MODELS_DIR}'."
        )


def _load_scaler(path: Path):
    """
    Load a scaler that was saved with np.save(...).
    Typically returns a 0-d object array -> .item().
    """
    _ensure_exists(path, "scaler")
    arr = np.load(path, allow_pickle=True)

    # If it's already a scaler-like object
    if hasattr(arr, "transform"):
        return arr

    # If it's a 0-d array wrapping the scaler
    if isinstance(arr, np.ndarray) and arr.shape == () and hasattr(arr.item(), "transform"):
        return arr.item()

    return arr


def load_v3_models():
    """
    Load and cache artifacts for Prophet + LSTM hybrid (v3).

    Returns:
        (lstm_model, x_scaler, y_scaler, look_back)
    """
    # LSTM model
    if _cache["v3_lstm"] is None:
        _ensure_exists(V3_PATHS["lstm"], "v3 LSTM model")
        _cache["v3_lstm"] = load_model(V3_PATHS["lstm"])

    # Scalers
    if _cache["v3_x_scaler"] is None:
        _cache["v3_x_scaler"] = _load_scaler(V3_PATHS["x_scaler"])

    if _cache["v3_y_scaler"] is None:
        _cache["v3_y_scaler"] = _load_scaler(V3_PATHS["y_scaler"])

    return (
        _cache["v3_lstm"],
        _cache["v3_x_scaler"],
        _cache["v3_y_scaler"],
        V3_LOOK_BACK,
    )


def load_v4_models():
    """
    Load and cache artifacts for SARIMAX + GRU hybrid (v4).

    Returns:
        (sarimax_model, gru_model, x_scaler, y_scaler, look_back)
    """
    # SARIMAX
    if _cache["v4_sarimax"] is None:
        _ensure_exists(V4_PATHS["sarimax"], "v4 SARIMAX model")
        _cache["v4_sarimax"] = joblib.load(V4_PATHS["sarimax"])

    # GRU
    if _cache["v4_gru"] is None:
        _ensure_exists(V4_PATHS["gru"], "v4 GRU model")
        _cache["v4_gru"] = load_model(V4_PATHS["gru"])

    # Scalers
    if _cache["v4_x_scaler"] is None:
        _cache["v4_x_scaler"] = _load_scaler(V4_PATHS["x_scaler"])

    if _cache["v4_y_scaler"] is None:
        _cache["v4_y_scaler"] = _load_scaler(V4_PATHS["y_scaler"])

    return (
        _cache["v4_sarimax"],
        _cache["v4_gru"],
        _cache["v4_x_scaler"],
        _cache["v4_y_scaler"],
        V4_LOOK_BACK,
    )
