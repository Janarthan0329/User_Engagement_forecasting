# backend/forecast/preprocessing.py
import pandas as pd

def preprocess_logs_to_timeseries(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Supports two input formats:

    1) Raw logs:
       - columns: 'timestamp' and 'user_id'
       -> aggregates to daily DAU.

    2) Already aggregated daily metrics (your logs.csv):
       - column: 'date'
       - and a DAU-like column: 'daily_active_users' or 'DAU' or 'dau'
       -> just parses and renames to 'DAU'.

    Returns a DataFrame indexed by 'date' with at least:
        ['DAU', 'day_of_week', 'month', 'is_weekend']
    """
    df = df_raw.copy()

    # --- CASE 1: raw logs with timestamp + user id ---
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")

        # normalize to midnight timestamps by using .dt.normalize() to keep dtype as datetime64[ns]
        df["date"] = df["timestamp"].dt.normalize() # type: ignore

        # use user_id if present, else treat each row as one active session
        if "user_id" in df.columns:
            dau = (
                df.groupby("date")["user_id"]
                .nunique()
                .rename("DAU")
                .reset_index()
            )
        else:
            # fallback: count rows per day
            dau = (
                df.groupby("date")
                .size()
                .rename("DAU")
                .reset_index()
            )

    # --- CASE 2: your aggregated CSV with 'date' column ---
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        # try to find the DAU column
        dau_col = None
        for cand in ["daily_active_users", "DAU", "dau", "daily_active_user"]:
            if cand in df.columns:
                dau_col = cand
                break

        if dau_col is None:
            raise ValueError(
                "No DAU column found. Expected one of: "
                "'daily_active_users', 'DAU', 'dau'."
            )

        dau = df[["date", dau_col]].rename(columns={dau_col: "DAU"}).copy()

    else:
        raise ValueError(
            "Input must contain either a 'timestamp' column (raw logs) "
            "or a 'date' column (aggregated daily data)."
        )

    # --- common feature engineering ---
    dau = dau.sort_values("date")
    dau["date"] = pd.to_datetime(dau["date"])
    # use DatetimeIndex to extract components so static type-checkers don't complain
    dau["day_of_week"] = pd.DatetimeIndex(dau["date"]).dayofweek
    dau["month"] = pd.DatetimeIndex(dau["date"]).month
    dau["is_weekend"] = dau["day_of_week"].isin([5, 6]).astype(int)

    dau.set_index("date", inplace=True)
    return dau
