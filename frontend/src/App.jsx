import React, { useState, useEffect } from "react";
import axios from "axios";

// Define the API URL as a constant
const API_URL = "http://localhost:5000";

// Remove the global axios defaults as they can cause CORS issues
// Configure axios for this specific instance instead

// Function to format disease names for display
const formatDiseaseName = (name) => {
  return name
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(" ");
};

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);

  // Check model status when component mounts
  useEffect(() => {
    checkModelStatus();
  }, []);

  const checkModelStatus = async () => {
    try {
      const response = await axios.get(`${API_URL}/`);
      setModelStatus(response.data.model_status);
      if (response.data.model_status !== "loaded") {
        setError("Model is not loaded. Please check server logs.");
      }
    } catch (error) {
      setError(
        "Could not connect to the server. Please ensure the backend is running."
      );
      console.error("Server connection error:", error);
    }
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
      setResult(null);
    }
  };

  const handleSubmit = async () => {
    if (!image) {
      setError("Please select an image first");
      return;
    }

    if (modelStatus !== "loaded") {
      setError("Model is not loaded. Please check server logs.");
      return;
    }

    const formData = new FormData();
    formData.append("image", image);
    setLoading(true);
    setError(null);

    try {
      console.log("Sending request to:", `${API_URL}/predict`);
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      console.log("Response received:", response.data);

      if (response.data.success) {
        setResult(response.data);
      } else {
        setError(response.data.error || "Prediction failed");
      }
    } catch (error) {
      console.error("Detailed error:", error);
      setError(
        error.response?.data?.error ||
          error.message ||
          "Failed to process the image. Please try again."
      );

      // If we get a 500 error, try to refresh the model status
      if (error.response?.status === 500) {
        await checkModelStatus();
      }
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.5) return "#2ecc71"; // High confidence - green
    if (confidence >= 0.3) return "#f1c40f"; // Medium confidence - yellow
    return "#e74c3c"; // Low confidence - red
  };

  const renderPredictions = (predictions) => {
    // Filter predictions with confidence > 15%
    const significantPredictions = predictions.filter(
      (pred) => pred.confidence > 0.15
    );

    // Sort by confidence
    significantPredictions.sort((a, b) => b.confidence - a.confidence);

    return (
      <div className="predictions-list">
        {significantPredictions.map((pred, index) => (
          <div
            key={index}
            className={`prediction-item ${index === 0 ? "top-prediction" : ""}`}
          >
            <div className="prediction-name">
              <span className="disease-name">
                {formatDiseaseName(pred.disease)}
              </span>
            </div>
            <div className="prediction-confidence">
              <div
                className="confidence-bar"
                style={{
                  width: `${pred.confidence * 100}%`,
                  backgroundColor: getConfidenceColor(pred.confidence),
                }}
              />
              <span
                className="confidence-value"
                style={{ color: getConfidenceColor(pred.confidence) }}
              >
                {(pred.confidence * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderImageFeatures = (features) => {
    if (!features) return null;

    return (
      <div className="image-features">
        <h4>Image Analysis:</h4>
        <div className="feature-stats">
          <div className="feature-item">
            <span className="feature-label">Blue Detection:</span>
            <span
              className={`feature-value ${
                features.is_bluish ? "blue-detected" : ""
              }`}
            >
              {features.is_bluish
                ? "Strong blue components"
                : "No significant blue"}
            </span>
          </div>
          <div className="feature-item">
            <span className="feature-label">Blue Ratio:</span>
            <span className="feature-value">
              {(features.blue_ratio * 100).toFixed(1)}%
            </span>
          </div>
          <div className="feature-item">
            <span className="feature-label">Color Uniformity:</span>
            <span className="feature-value">
              {features.color_variance < 0.1
                ? "High"
                : features.color_variance < 0.2
                ? "Medium"
                : "Low"}
            </span>
          </div>
          <div className="feature-item">
            <span className="feature-label">Texture:</span>
            <span className="feature-value">
              {features.texture_energy < 0.3
                ? "Smooth"
                : features.texture_energy < 0.6
                ? "Medium"
                : "Rough"}
            </span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="App">
      <h2>Skin Disease Detection</h2>

      {modelStatus && modelStatus !== "loaded" && (
        <div className="model-status-warning">
          <p>
            Warning: Model is not loaded properly. Some features may not work.
          </p>
        </div>
      )}

      <div className="upload-section">
        <input
          type="file"
          onChange={handleImageChange}
          accept="image/*"
          className="file-input"
        />
        {previewUrl && (
          <div className="image-preview">
            <img
              src={previewUrl}
              alt="Preview"
              style={{ maxWidth: "300px", marginTop: "10px" }}
            />
          </div>
        )}
        <button
          onClick={handleSubmit}
          disabled={loading || !image || modelStatus !== "loaded"}
          className={`submit-button ${loading ? "loading" : ""}`}
        >
          {loading ? "Processing..." : "Submit"}
        </button>
      </div>

      {error && (
        <div className="error-message">
          <p>{error}</p>
          <p className="help-text">
            If you're seeing a connection error, please ensure the backend
            server is running on port 5000.
          </p>
        </div>
      )}

      {result && (
        <div className="result-section">
          <h3>Prediction Result:</h3>
          {renderPredictions(result.all_predictions)}
          <div className="prediction-note">
            <p>Note: Only showing predictions with confidence above 15%</p>
          </div>
          <p className="warning-text">
            Note: Please consult a healthcare professional for accurate
            diagnosis. These predictions are for educational purposes only.
          </p>
        </div>
      )}

      <div className="disclaimer">
        <p>
          Disclaimer: This tool is for educational purposes only and should not
          be used as a substitute for professional medical advice. Always
          consult with a qualified healthcare provider for proper diagnosis and
          treatment.
        </p>
      </div>
    </div>
  );
}

export default App;
