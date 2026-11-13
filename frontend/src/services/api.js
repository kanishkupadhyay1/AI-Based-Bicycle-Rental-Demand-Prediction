import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const fetchHealth = async () => {
  const { data } = await apiClient.get('/');
  return data;
};

export const fetchMetrics = async () => {
  const { data } = await apiClient.get('/api/metrics');
  return data;
};

export const fetchModelComparison = async () => {
  const { data } = await apiClient.get('/api/model-comparison');
  return data;
};

export const fetchDemandAnalysis = async () => {
  const { data } = await apiClient.get('/api/demand-analysis');
  return data;
};

export const fetchGamEffects = async () => {
  const { data } = await apiClient.get('/api/gam-effects');
  return data;
};

export const fetchFeatureInfo = async () => {
  const { data } = await apiClient.get('/api/feature-info');
  return data;
};

export const predictSingleModel = async (payload) => {
  const { data } = await apiClient.post('/api/predict', payload);
  return data;
};

export const predictAllModels = async (payload) => {
  const { data } = await apiClient.post('/api/predict-all', payload);
  return data;
};

export default apiClient;
