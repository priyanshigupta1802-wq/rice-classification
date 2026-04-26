import { useState, useRef } from "react";
import "./App.css";

// ← Your PC's local network IP so mobile can reach Flask
const API_URL = "http://192.168.1.8:5000/predict";

const RICE_EMOJIS = {
  Arborio: "🍚",
  Basmati: "🌾",
  Ipsala: "🌿",
  Jasmine: "🌸",
  Karacadag: "🌱",
};

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  const handleCapture = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setResult(null);
      setError("");
      setPreview(URL.createObjectURL(file));
    }
  };

  const handlePredict = async () => {
    if (!image) return;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", image);

      const res = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error(`Server error ${res.status}`);

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(`Could not reach the server. Make sure you're on the same Wi-Fi and app.py is running.\n\n${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  };

  const confidence = result ? Math.round(result.confidence * 100) : 0;

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-icon">🌾</div>
        <h1 className="header-title">Rice Classifier</h1>
        <p className="header-sub">Identify rice varieties instantly</p>
      </header>

      <main className="main">

        {/* Upload Zone */}
        <div
          className={`upload-zone ${preview ? "has-image" : ""}`}
          onClick={() => inputRef.current?.click()}
        >
          {preview ? (
            <img src={preview} alt="Selected rice" className="preview-img" />
          ) : (
            <div className="upload-placeholder">
              <span className="upload-icon">📷</span>
              <p className="upload-text">Tap to take a photo<br /><span>or choose from gallery</span></p>
            </div>
          )}
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            capture="environment"
            onChange={handleCapture}
            style={{ display: "none" }}
          />
        </div>

        {/* Actions */}
        <div className="actions">
          {preview && (
            <button className="btn btn-secondary" onClick={handleReset}>
              ↺ Reset
            </button>
          )}
          <button
            className={`btn btn-primary ${loading ? "loading" : ""}`}
            onClick={handlePredict}
            disabled={!image || loading}
          >
            {loading ? (
              <><span className="spinner" /> Analyzing...</>
            ) : (
              "🔍 Predict"
            )}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="error-card">
            <span className="error-icon">⚠️</span>
            <p>{error}</p>
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="result-card">
            <div className="result-emoji">
              {RICE_EMOJIS[result.class] || "🌾"}
            </div>
            <h2 className="result-class">{result.class}</h2>
            <p className="result-label">Rice Variety</p>

            {/* Top confidence bar */}
            <div className="confidence-section">
              <div className="confidence-header">
                <span>Confidence</span>
                <span className="confidence-pct">{confidence}%</span>
              </div>
              <div className="confidence-bar">
                <div
                  className="confidence-fill"
                  style={{ width: `${confidence}%` }}
                />
              </div>
            </div>

            {/* All class breakdown */}
            {result.all_scores && (
              <div className="breakdown">
                <p className="breakdown-title">All Classes</p>
                {Object.entries(result.all_scores)
                  .sort((a, b) => b[1] - a[1])
                  .map(([cls, score]) => {
                    const pct = Math.round(score * 100);
                    const isWinner = cls === result.class;
                    return (
                      <div key={cls} className="breakdown-row">
                        <span className={`breakdown-label ${isWinner ? "winner" : ""}`}>
                          {RICE_EMOJIS[cls]} {cls}
                        </span>
                        <div className="breakdown-bar-wrap">
                          <div
                            className={`breakdown-bar-fill ${isWinner ? "winner" : ""}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className={`breakdown-pct ${isWinner ? "winner" : ""}`}>{pct}%</span>
                      </div>
                    );
                  })}
              </div>
            )}
          </div>
        )}

        {/* Network hint */}
        <p className="network-hint">
          📡 Make sure your phone and PC are on the same Wi-Fi network
        </p>
      </main>
    </div>
  );
}

export default App;