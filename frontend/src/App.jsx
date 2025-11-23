import React, { useEffect, useState } from "react";
import Landing from "./pages/Landing";
import ForcastingPage from "./pages/ForcastingPage";

export default function App() {
  const [started, setStarted] = useState(false);
  const [backendStatus, setBackendStatus] = useState("Checking...");
  const [file, setFile] = useState(null);
  const [horizon, setHorizon] = useState(14);
  const [modelType, setModelType] = useState("v4");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState(""); // NEW
  const [errorMsg, setErrorMsg] = useState("");
  const [startDate, setStartDate] = useState("");

  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/user-engagement/")
      .then((res) => res.json())
      .then((data) => setBackendStatus(`${data.status}: ${data.message}`))
      .catch(() => setBackendStatus("Error connecting to backend"));
  }, []);

  const handleFileChange = (e) => {
    setFile(e.target.files[0] || null);
  };

  const handleSubmit = async (e) => {
    e?.preventDefault?.();
    setErrorMsg("");
    setResult(null);
    if (!file) {
      setErrorMsg("Please select a CSV file first.");
      return;
    }

    setLoading(true);
    setLoadingStage("Processing your data...");
    const formData = new FormData();
    formData.append("file", file);
    formData.append("horizon", String(horizon));
    formData.append("model_type", modelType);
    formData.append("start_date", startDate || "");

    try {
      // Simulate step 1: processing data
      setLoadingStage("Processing your data...");
      await new Promise((r) => setTimeout(r, 400)); // optional: fake delay

      // Simulate step 2: generating forecast
      setLoadingStage("Generating forecast...");
      await new Promise((r) => setTimeout(r, 400)); // optional: fake delay

      // Simulate step 3: generating plots
      setLoadingStage("Generating plots and graph...");
      // Actually send request
      const res = await fetch("http://127.0.0.1:8000/api/forecast/", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `Request failed: ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
      setLoadingStage("Ready to forecast!"); // Final stage
    } catch (err) {
      setErrorMsg(err.message || String(err));
      setLoadingStage("");
    } finally {
      setLoading(false);
      setTimeout(() => setLoadingStage(""), 1000); // Clear after short delay
    }
  };

  if (!started) {
    return <Landing onStart={() => setStarted(true)} />;
  }

  return (
    <ForcastingPage
      backendStatus={backendStatus}
      horizon={horizon}
      setHorizon={setHorizon}
      modelType={modelType}
      setModelType={setModelType}
      result={result}
      loading={loading}
      loadingStage={loadingStage} // NEW
      errorMsg={errorMsg}
      startDate={startDate}
      setStartDate={setStartDate}
      handleFileChange={handleFileChange}
      handleSubmit={handleSubmit}
    />
  );
}