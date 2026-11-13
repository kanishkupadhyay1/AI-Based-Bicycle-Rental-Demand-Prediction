import React, { useEffect, useState } from 'react';
import { fetchFeatureInfo, predictAllModels, predictSingleModel } from '../services/api';
import PredictionForm from '../components/PredictionForm';
import PredictionResult from '../components/PredictionResult';

export default function Prediction() {
  const [featureInfo, setFeatureInfo] = useState(null);
  const [singleResult, setSingleResult] = useState(null);
  const [allResults, setAllResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    fetchFeatureInfo()
      .then((data) => {
        if (active) setFeatureInfo(data);
      })
      .catch((err) => {
        if (active) setError(err.response?.data?.detail || err.message);
      });
    return () => {
      active = false;
    };
  }, []);

  const handlePredictSingle = async (payload) => {
    try {
      setLoading(true);
      setError(null);
      const res = await predictSingleModel(payload);
      setSingleResult(res);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Prediction failed.');
    } finally {
      setLoading(false);
    }
  };

  const handlePredictAll = async (payload) => {
    try {
      setLoading(true);
      setError(null);
      const res = await predictAllModels(payload);
      setAllResults(res);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Multi-model prediction failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-slate-900">Real-Time Bicycle Demand Prediction</h2>
        <p className="text-sm text-slate-600">
          Submit weather and temporal conditions to the FastAPI backend to compute predictions
          using the persisted trained models.
        </p>
      </div>

      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {error}
        </div>
      )}

      <PredictionForm
        featureInfo={featureInfo}
        onPredictSingle={handlePredictSingle}
        onPredictAll={handlePredictAll}
        loading={loading}
      />

      <PredictionResult singleResult={singleResult} allResults={allResults} />
    </div>
  );
}
