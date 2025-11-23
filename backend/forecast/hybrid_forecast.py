# backend/forecast/hybrid_forecast.py
from typing import Literal, Tuple
import numpy as np
import pandas as pd
from prophet import Prophet

from .model_loader import load_v3_models, load_v4_models


# ================================================================
#  Training distribution (from original Excel dataset)
# ================================================================
TRAIN_DAU_MEAN = 1194.085665
TRAIN_DAU_STD = 1196.521192


def _build_sequence(series: np.ndarray, look_back: int) -> np.ndarray:
    series = np.asarray(series).flatten()
    if len(series) < look_back:
        pad_len = look_back - len(series)
        series = np.concatenate([np.repeat(series[0], pad_len), series])
    else:
        series = series[-look_back:]
    return series.reshape(1, look_back, 1)


def _scale(data: np.ndarray, scaler, inverse: bool = False) -> np.ndarray:
    arr = np.asarray(data).reshape(-1, 1)
    if hasattr(scaler, "inverse_transform") and inverse:
        out = scaler.inverse_transform(arr)
    elif hasattr(scaler, "transform") and not inverse:
        out = scaler.transform(arr)
    else:
        out = arr
    return out.flatten()


def _match_model_input(model, x: np.ndarray) -> np.ndarray:
    try:
        input_shape = model.input_shape
    except Exception:
        return x

    if not input_shape:
        return x

    if len(input_shape) == 3:
        _, exp_timesteps, exp_features = input_shape
    elif len(input_shape) == 2:
        _, exp_features = input_shape
        exp_timesteps = 1
    else:
        return x

    if exp_timesteps is None or exp_features is None:
        return x

    if x.shape[1] == exp_timesteps and x.shape[2] == exp_features:
        return x

    if x.shape[2] == 1 and exp_features > 1:
        return np.tile(x, (1, 1, exp_features))

    total = x.shape[1] * x.shape[2]
    if total == (exp_timesteps * exp_features):
        return x.reshape((x.shape[0], exp_timesteps, exp_features))

    raise ValueError(
        f"Model expects input shape (batch,{exp_timesteps},{exp_features}) but got {x.shape}."
    )


# ===================================================================
# V3 — PROPHET + LSTM (Residual Multiplicative Hybrid)
# ===================================================================

def forecast_v3_prophet_lstm(ts_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    ts_df: index = date, must contain column 'DAU' in TRAINING SCALE.
    The rescaling from new dataset space -> training space is done externally.
    """
    lstm_model, x_scaler, y_scaler, look_back = load_v3_models()

    dau_df = ts_df[["DAU"]].reset_index().rename(columns={"index": "date"})
    dau_df.rename(columns={"date": "ds", "DAU": "y"}, inplace=True)
    dau_df["ds"] = pd.to_datetime(dau_df["ds"])
    last_date = dau_df["ds"].max()

    prophet = Prophet(
        seasonality_mode="multiplicative",
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
    )
    prophet.fit(dau_df)

    future = prophet.make_future_dataframe(periods=horizon)
    forecast = prophet.predict(future)

    hist_forecast = forecast[forecast["ds"] <= last_date][["ds", "yhat"]]
    merged = dau_df.merge(hist_forecast, on="ds", how="inner")

    merged["yhat_adj"] = merged["yhat"].replace(0, 1e-8)
    merged["ratio"] = merged["y"] / merged["yhat_adj"]
    ratio_series = merged["ratio"].to_numpy(dtype=float)

    ratio_scaled = _scale(np.asarray(ratio_series), x_scaler, inverse=False)
    x_input = _build_sequence(ratio_scaled, look_back)

    x_input = _match_model_input(lstm_model, x_input)
    nn_raw = lstm_model.predict(x_input)[0]
    nn_raw = np.asarray(nn_raw).flatten()

    if len(nn_raw) < horizon:
        nn_raw = np.concatenate([nn_raw, np.repeat(nn_raw[-1], horizon - len(nn_raw))])

    ratio_future = _scale(nn_raw[:horizon], y_scaler, inverse=True)

    future_fc = forecast[forecast["ds"] > last_date].head(horizon).copy()
    future_fc["ratio_future"] = ratio_future

    # multiplicative hybrid in TRAINING SCALE
    future_fc["forecast_DAU"] = future_fc["yhat"] * future_fc["ratio_future"]

    out = future_fc[["ds", "forecast_DAU"]].copy()
    out.rename(columns={"ds": "date"}, inplace=True)
    out["date"] = pd.to_datetime(out["date"])
    out.set_index("date", inplace=True)
    return out


# ===================================================================
# V4 — SARIMAX + GRU (Additive Hybrid)
# ===================================================================

def _expected_k_exog(sarimax_model) -> int | None:
    k = getattr(sarimax_model, "k_exog", None)
    if k is not None and k != 0:
        return k
    try:
        k = getattr(getattr(sarimax_model, "model", None), "k_exog", None)
        if k is not None and k != 0:
            return k
    except Exception:
        pass
    try:
        names = getattr(getattr(sarimax_model, "model", None), "exog_names", None)
        if names:
            return len(names)
    except Exception:
        pass
    return None


def forecast_v4_sarimax_gru(ts_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    ts_df: index = date, must contain column 'DAU' in TRAINING SCALE.
    """
    sarimax_model, gru_model, x_scaler, y_scaler, look_back = load_v4_models()

    series = ts_df["DAU"].astype(float)
    series = series.sort_index()

    last_date = series.index.max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    # base exog (we'll pad/truncate to k_exog)
    exog_base = pd.DataFrame(
        {
            "day_of_week": future_dates.dayofweek,
            "month": future_dates.month,
            "is_weekend": future_dates.dayofweek.isin([5, 6]).astype(int),
        },
        index=future_dates,
    ).astype(float)

    expected_k = _expected_k_exog(sarimax_model)

    if expected_k is not None and expected_k > 0:
        exog_future = exog_base.copy()
        cur_k = exog_future.shape[1]
        if cur_k < expected_k:
            for i in range(cur_k, expected_k):
                exog_future[f"pad_{i}"] = 0.0
        elif cur_k > expected_k:
            exog_future = exog_future.iloc[:, :expected_k]
    else:
        exog_future = None

    # 1. SARIMAX forecast in TRAINING SCALE
    if exog_future is not None:
        sarimax_fc_res = sarimax_model.get_forecast(steps=horizon, exog=exog_future)
    else:
        sarimax_fc_res = sarimax_model.get_forecast(steps=horizon)

    sarimax_pred = getattr(sarimax_fc_res, "predicted_mean", sarimax_fc_res)
    sarimax_future_mean = np.asarray(sarimax_pred).flatten()

    if sarimax_future_mean.size < horizon:
        if sarimax_future_mean.size == 0:
            sarimax_future_mean = np.zeros(horizon)
        else:
            sarimax_future_mean = np.pad(
                sarimax_future_mean,
                (0, horizon - sarimax_future_mean.size),
                mode="edge",
            )[:horizon]

    # 2. In-sample residuals (TRAINING SCALE)
    in_sample = sarimax_model.get_prediction()
    fitted_raw = getattr(in_sample, "predicted_mean", None)

    aligned = series.to_frame(name="DAU").copy()

    if isinstance(fitted_raw, pd.Series):
        fitted_s = fitted_raw.reindex(aligned.index)
    else:
        arr = np.asarray(fitted_raw).flatten()
        if len(arr) == len(aligned.index):
            fitted_s = pd.Series(arr, index=aligned.index)
        elif len(arr) < len(aligned.index):
            fitted_s = pd.Series(arr, index=aligned.index[-len(arr):]).reindex(aligned.index)
        else:
            fitted_s = pd.Series(arr[-len(aligned.index):], index=aligned.index)

    aligned["sarimax_hat"] = fitted_s
    aligned.dropna(inplace=True)

    aligned["residual"] = aligned["DAU"] - aligned["sarimax_hat"]
    residual_series = aligned["residual"].to_numpy(dtype=float)

    res_scaled = _scale(np.asarray(residual_series), x_scaler, inverse=False)
    x_input = _build_sequence(res_scaled, look_back)

    x_input = _match_model_input(gru_model, x_input)
    nn_raw = gru_model.predict(x_input)[0]
    nn_raw = np.asarray(nn_raw).flatten()

    if len(nn_raw) < horizon:
        nn_raw = np.concatenate([nn_raw, np.repeat(nn_raw[-1], horizon - len(nn_raw))])

    residual_future = _scale(nn_raw[:horizon], y_scaler, inverse=True)

    # additive hybrid in TRAINING SCALE
    final_forecast = sarimax_future_mean + residual_future

    out = pd.DataFrame({"forecast_DAU": final_forecast}, index=future_dates)
    return out


# ===================================================================
# Router with distribution mapping
# ===================================================================

def run_hybrid_forecast(
    ts_df: pd.DataFrame,
    horizon: int,
    model_type: Literal["v3", "v4"] = "v4",
) -> Tuple[pd.DataFrame, str]:
    """
    1. Take DAU in NEW DATASET SCALE (from preprocessing).
    2. Map it into TRAINING SCALE.
    3. Run chosen hybrid model (v3 or v4).
    4. Map forecast back to NEW DATASET SCALE.
    5. Enforce DAU >= 0.
    """
    # --- 1) Original series in new dataset space ---
    ts_df = ts_df.copy()
    orig_series = ts_df["DAU"].astype(float)
    orig_mean = float(orig_series.mean())
    orig_std = float(orig_series.std()) or 1.0
    if orig_std < 1e-6:
        orig_std = 1.0

    # --- 2) Map into training distribution ---
    train_std = TRAIN_DAU_STD if TRAIN_DAU_STD > 1e-6 else 1.0
    train_mean = TRAIN_DAU_MEAN

    dau_train_space = (orig_series - orig_mean) / orig_std * train_std + train_mean
    ts_train = ts_df.copy()
    ts_train["DAU"] = dau_train_space

    # --- 3) Run hybrid in training space ---
    if model_type == "v3":
        fc_train = forecast_v3_prophet_lstm(ts_train, horizon)
        label = "Prophet + LSTM Residual Multiplicative Hybrid (v3, win=28)"
    elif model_type == "v4":
        fc_train = forecast_v4_sarimax_gru(ts_train, horizon)
        label = "SARIMAX + GRU Ensemble Hybrid (v4, win=28)"
    else:
        raise ValueError("Unknown model_type (use 'v3' or 'v4').")

    # --- 4) Map forecast back to new dataset scale ---
    pred_train = fc_train["forecast_DAU"].astype(float)
    pred_new = (pred_train - train_mean) / train_std * orig_std + orig_mean

    # --- 5) Enforce non-negative DAU in output space ---
    pred_new = np.maximum(pred_new, 0.0)

    fc_out = fc_train.copy()
    fc_out["forecast_DAU"] = pred_new

    # Normalise index for JSON
    idx = pd.DatetimeIndex(fc_out.index)
    fc_out.index = idx.to_period("D").to_timestamp()

    return fc_out, label
