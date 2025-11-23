# backend/forecast/plotting.py
import pandas as pd
import matplotlib.pyplot as plt
import base64, io
from prophet import Prophet

def generate_forecast_plot(history_df, forecast_df, model_label):
    """
    Returns a base64 PNG string of (history + forecast) line plot.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    # HISTORY
    ax.plot(
        history_df.index,
        history_df["DAU"],
        label="History",
        color="#4ade80",   # green
        linewidth=2,
    )

    # FORECAST
    ax.plot(
        forecast_df.index,
        forecast_df["forecast_DAU"],
        label="Forecast",
        color="#60a5fa",   # blue
        linewidth=2,
    )

    # LAST HISTORY POINT MARKER
    ax.scatter(
        history_df.index[-1],
        history_df["DAU"].iloc[-1],
        color="white",
        edgecolors="black",
        s=60,
        zorder=3,
    )

    ax.set_title(f"Forecast Result — {model_label}", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Active Users (DAU)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Convert figure to base64 PNG
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)

    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return encoded



def generate_forecast_zoom_plot(forecast_df, model_label):
    """
    Returns a base64 PNG string of a zoomed-in forecast-only plot
    over the forecast date range.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(
        forecast_df.index,
        forecast_df["forecast_DAU"],
        label="Forecast (zoomed)",
        linewidth=2,
    )

    ax.set_title(f"Zoomed Forecast — {model_label}", fontsize=12)
    ax.set_xlabel("Forecast Date")
    ax.set_ylabel("Daily Active Users (DAU)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)

    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return encoded




def generate_prophet_components_plot(history_df, horizon, model_label):
    """
    Fit a standalone Prophet model on the DAU history and return
    the trend/seasonality components plot as base64 PNG.
    
    history_df: DataFrame with index = date, column "DAU"
    """
    # Prepare data for Prophet
    df = history_df.reset_index().rename(columns={"index": "date"})
    df.rename(columns={"date": "ds", "DAU": "y"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"])

    # Same seasonality mode as hybrid v3 (multiplicative)
    m = Prophet(
        seasonality_mode="multiplicative",
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
    )
    m.fit(df)

    future = m.make_future_dataframe(periods=horizon)
    forecast = m.predict(future)

    # Create components figure
    fig = m.plot_components(forecast)
    fig.suptitle(f"Prophet Components — {model_label}", fontsize=12)

    # Encode figure as base64 PNG
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)

    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return encoded