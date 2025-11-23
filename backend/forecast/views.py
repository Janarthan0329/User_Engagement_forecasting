# backend/forecast/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.dateparse import parse_date

import pandas as pd
from datetime import timedelta

from .preprocessing import preprocess_logs_to_timeseries
from .hybrid_forecast import run_hybrid_forecast
from .plotting import (
    generate_forecast_plot,
    generate_forecast_zoom_plot,
    generate_prophet_components_plot,   # ✅ NEW
)


def user_engagement_check(request):
    return JsonResponse({
        "status": "ok",
        "message": "Forecasting backend is up",
    })


@csrf_exempt
def forecast_from_csv(request):
    """
    POST /api/forecast/
    multipart/form-data:
      - file: CSV (logs.csv or raw logs)
      - horizon (optional)
      - model_type (optional) 'v3'|'v4'
      - start_date (optional)
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    if "file" not in request.FILES:
        return JsonResponse({"error": "CSV file is required (field name 'file')"}, status=400)

    csv_file = request.FILES["file"]
    try:
        df_raw = pd.read_csv(csv_file)
    except Exception as e:
        return JsonResponse({"error": f"Failed to read CSV: {e}"}, status=400)

    # horizon
    try:
        horizon = int(request.POST.get("horizon", 14))
    except ValueError:
        horizon = 14

    # model selection
    model_type = request.POST.get("model_type", "v4")

    # 1. Preprocess -> DAU time series
    try:
        ts_df = preprocess_logs_to_timeseries(df_raw)
    except Exception as e:
        return JsonResponse({"error": f"Preprocessing failed: {e}"}, status=400)

    if ts_df.empty:
        return JsonResponse({"error": "No valid data after preprocessing."}, status=400)

    # 2. Hybrid forecast
    try:
        forecast_df, model_label = run_hybrid_forecast(ts_df, horizon, model_type=model_type)
    except Exception as e:
        return JsonResponse({"error": f"Forecasting failed: {e}"}, status=500)

    # Optional custom forecast start date
    start_date_str = request.POST.get("start_date")
    if start_date_str:
        parsed = parse_date(start_date_str)
        if parsed is None:
            return JsonResponse({"error": "Invalid start_date; expected YYYY-MM-DD"}, status=400)
        try:
            new_index = pd.date_range(start=pd.to_datetime(parsed), periods=len(forecast_df), freq="D")
            forecast_df = forecast_df.copy()
            forecast_df.index = new_index
        except Exception as e:
            return JsonResponse({"error": f"Failed to apply start_date: {e}"}, status=400)

    # 3. Generate plots
    try:
        plot_base64 = generate_forecast_plot(
            ts_df[["DAU"]],
            forecast_df[["forecast_DAU"]],
            model_label,
        )
        plot_zoom_base64 = generate_forecast_zoom_plot(
            forecast_df[["forecast_DAU"]],
            model_label,
        )
    except Exception:
        plot_base64 = None
        plot_zoom_base64 = None

    # 4. Prophet components (only meaningful for v3 / Prophet hybrid)
    prophet_components_base64 = None
    if model_type == "v3":
        try:
            prophet_components_base64 = generate_prophet_components_plot(
                ts_df[["DAU"]],
                horizon,
                model_label,
            )
        except Exception:
            prophet_components_base64 = None

    # history
    history_rows = [
        {"date": pd.to_datetime(str(idx)).strftime("%Y-%m-%d"), "DAU": float(row["DAU"])}
        for idx, row in ts_df[["DAU"]].iterrows()
    ]

    # forecast
    forecast_rows = [
        {"date": pd.to_datetime(str(idx)).strftime("%Y-%m-%d"), "forecast_DAU": float(row["forecast_DAU"])}
        for idx, row in forecast_df[["forecast_DAU"]].iterrows()
    ]

    return JsonResponse({
        "horizon": horizon,
        "model_type": model_type,
        "model_label": model_label,
        "history": history_rows,
        "forecast": forecast_rows,
        "plot_base64": plot_base64,
        "plot_zoom_base64": plot_zoom_base64,
        "prophet_components_base64": prophet_components_base64,  # ✅ NEW
    })
