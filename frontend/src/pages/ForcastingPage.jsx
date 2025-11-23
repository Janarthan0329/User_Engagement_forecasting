import React, { useState, useEffect } from "react";
import "./ForcastingPage.css";

export default function ForcastingPage({
    backendStatus,
    horizon,
    setHorizon,
    modelType,
    setModelType,
    result,
    loading,
    loadingStage, 
    errorMsg,
    startDate,
    setStartDate,
    handleFileChange,
    handleSubmit,
}) {

    const [showModal, setShowModal] = useState(false);

    useEffect(() => {
        function onKey(e) {
            if (e.key === "Escape") setShowModal(false);
        }
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, []);

    const columnsText =
        "date, application_name, daily_active_users, session_count, avg_session_duration, like_count, share_count, download_count, comment_count, error_count, day_of_week, month, is_weekend, release_flag";

    return (
        <div className="fp-root">
            <aside className="fp-sidebar">
                <h1 className="fp-title">User Engagement Forecast Tool</h1>
                <p className="fp-status">Backend status: {backendStatus}</p>

                <form onSubmit={handleSubmit}>
                    <h2 className="fp-h2">Upload Usage Logs (CSV)</h2>

                    <div className="fp-field">
                        <label>CSV File:</label>
                        <input type="file" accept=".csv" onChange={handleFileChange} />
                        <div style={{ marginTop: 8 }}>
                            <button
                                type="button"
                                className="fp-link"
                                onClick={() => setShowModal(true)}
                                aria-haspopup="dialog"
                            >
                                Accepted columns (click to view)
                            </button>
                        </div>
                    </div>

                    <div className="fp-field">
                        <label>Forecast horizon (days):</label>
                        <input
                            type="number"
                            min="1"
                            max="60"
                            value={horizon}
                            onChange={(e) => setHorizon(Number(e.target.value))}
                            className="fp-input-number"
                        />
                    </div>

                    <div className="fp-field">
                        <label>Forecast start date (optional):</label>
                        <input
                            type="date"
                            value={startDate}
                            onChange={(e) => setStartDate(e.target.value)}
                            className="fp-input-date"
                        />
                    </div>

                    <div className="fp-field">
                        <label>Forecasting model:</label>
                        <select
                            value={modelType}
                            onChange={(e) => setModelType(e.target.value)}
                            className="fp-select"
                        >
                            <option value="v3">Prophet + LSTM (v3)</option>
                            <option value="v4">SARIMAX + GRU (v4)</option>
                        </select>
                    </div>

                    {errorMsg && <p className="fp-error">{errorMsg}</p>}

                    <div className="fp-actions">
                        <button type="submit" disabled={loading} className="fp-btn">
                            {loading ? "Processing..." : "Generate Forecast"}
                        </button>
                    </div>
                </form>
            </aside>

            <main className="fp-main">
                {loading ? (
                    <div className="fp-loading">
                        <div className="fp-loading-spinner"></div>
                        <h2 style={{ marginTop: 16, color: "#60a5fa" }}>
                            {loadingStage || "Processing..."}
                        </h2>
                    </div>
                ) : !result ? (
                    <div className="fp-empty">
                        <div>
                            <h2>Ready to generate a forecast</h2>
                            <p>Upload your CSV on the left and click "Generate Forecast".</p>
                        </div>
                    </div>
                ) : (
                    <div className="fp-result">
                        <h2>Forecast Result (horizon: {result.horizon} days)</h2>
                        <p className="fp-model">
                            Model used: <strong>{result.model_label}</strong>
                        </p>

                        {/* Chart 1: History + Forecast */}
                        {result.plot_base64 && (
                            <div className="fp-chart-section">
                                <h3>Forecast Chart (History + Forecast)</h3>
                                <img
                                    src={`data:image/png;base64,${result.plot_base64}`}
                                    alt="Forecast chart"
                                    className="fp-chart-image"
                                />
                            </div>
                        )}

                        {/* Chart 2: Zoomed Forecast */}
                        {result.plot_zoom_base64 && (
                            <div className="fp-chart-section">
                                <h3>Zoomed Forecast (Selected Period)</h3>
                                <img
                                    src={`data:image/png;base64,${result.plot_zoom_base64}`}
                                    alt="Zoomed forecast chart"
                                    className="fp-chart-image"
                                />
                            </div>
                        )}

                        {/* Chart 3: Prophet trend & seasonality components (v3 only) */}
                        {result.prophet_components_base64 && (
                            <div className="fp-chart-section">
                                <h3>Prophet Trend & Seasonality Components</h3>
                                <img
                                    src={`data:image/png;base64,${result.prophet_components_base64}`}
                                    alt="Prophet components chart"
                                    className="fp-chart-image"
                                />
                            </div>
                        )}

                        <h3>Forecast (preview)</h3>
                        <table className="fp-table">
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Forecast DAU</th>
                                </tr>
                            </thead>
                            <tbody>
                                {result.forecast.slice(0, 50).map((row) => (
                                    <tr key={row.date}>
                                        <td>{row.date}</td>
                                        <td>{Math.round(row.forecast_DAU)}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>

                        <p className="fp-note">
                            Showing first 50 forecast days. Full series is in the API response.
                        </p>
                    </div>
                )}
            </main>

            {showModal && (
                <div
                    className="fp-modal-overlay"
                    role="dialog"
                    aria-modal="true"
                    onClick={(e) => {
                        if (e.target === e.currentTarget) setShowModal(false);
                    }}
                >
                    <div className="fp-modal-content" role="document">
                        <button
                            className="fp-modal-close"
                            onClick={() => setShowModal(false)}
                            aria-label="Close"
                        >
                            ×
                        </button>
                        <h3>Accepted CSV columns</h3>
                        <pre className="fp-pre" style={{ marginTop: 8 }}>
                            {columnsText}
                        </pre>
                    </div>
                </div>
            )}
        </div>
    );
}
