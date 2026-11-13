import React, { useEffect, useState } from 'react';
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { fetchModelComparison } from '../services/api';
import ModelComparisonChart from '../components/ModelComparisonChart';

export default function ModelAnalysis() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    fetchModelComparison()
      .then((res) => {
        if (active) {
          setData(res);
          setError(null);
        }
      })
      .catch((err) => {
        if (active) setError(err.response?.data?.detail || err.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  if (loading) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-10 text-center text-sm text-slate-500">
        Loading model evaluation results...
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-6 text-sm text-red-700">
        {error}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-blue-200 bg-blue-50/70 p-5">
        <h2 className="text-lg font-bold text-slate-900">
          Model Evaluation &amp; Comparison (Chronological 20% Test Set)
        </h2>
        <p className="mt-1 text-sm text-slate-700">
          Best model by MAE: <span className="font-semibold">{data.best_model_by_mae}</span> &bull;
          Best model by RMSE: <span className="font-semibold">{data.best_model_by_rmse}</span>
        </p>
      </div>

      <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
        <table className="min-w-full divide-y divide-slate-200 text-sm">
          <thead className="bg-slate-50 text-left text-xs font-semibold uppercase text-slate-600">
            <tr>
              <th className="px-5 py-3">Model</th>
              <th className="px-5 py-3">MAE</th>
              <th className="px-5 py-3">RMSE</th>
              <th className="px-5 py-3">Poisson Deviance Explained</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {data.metrics.map((row) => (
              <tr key={row.model} className="hover:bg-blue-50/40">
                <td className="px-5 py-3 font-medium text-slate-900">{row.model}</td>
                <td className="px-5 py-3 font-mono">{row.mae.toFixed(4)}</td>
                <td className="px-5 py-3 font-mono">{row.rmse.toFixed(4)}</td>
                <td className="px-5 py-3 font-mono">{(row.deviance_explained * 100).toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <ModelComparisonChart metrics={data.metrics} metricKey="both" />
        <ModelComparisonChart metrics={data.metrics} metricKey="deviance_explained" />
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <h3 className="mb-1 text-base font-semibold text-slate-900">
          Actual vs Predicted Scatter (Test Set Sample)
        </h3>
        <p className="mb-4 text-xs text-slate-500">
          Comparison of predicted hourly counts against true observed counts (cnt)
        </p>
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis type="number" dataKey="actual_cnt" name="Actual Count" tick={{ fontSize: 12 }} />
              <YAxis type="number" dataKey="gam_pred" name="Predicted Count" tick={{ fontSize: 12 }} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Legend />
              <Scatter name="GAM" data={data.scatter_points} fill="#2563eb" fillOpacity={0.55} />
              <Scatter
                name="Gradient Boosting"
                data={data.scatter_points.map((p) => ({
                  actual_cnt: p.actual_cnt,
                  gam_pred: p.gradient_boosting_pred,
                }))}
                fill="#059669"
                fillOpacity={0.55}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
        <div className="border-b border-slate-100 px-5 py-3">
          <h3 className="text-base font-semibold text-slate-900">Residual Analysis Summary</h3>
        </div>
        <table className="min-w-full divide-y divide-slate-200 text-sm">
          <thead className="bg-slate-50 text-left text-xs font-semibold uppercase text-slate-600">
            <tr>
              <th className="px-5 py-3">Model</th>
              <th className="px-5 py-3">Mean Residual</th>
              <th className="px-5 py-3">Std Residual</th>
              <th className="px-5 py-3">Median |Error|</th>
              <th className="px-5 py-3">90th Percentile |Error|</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {data.residual_stats.map((row) => (
              <tr key={row.model}>
                <td className="px-5 py-3 font-medium text-slate-900">{row.model}</td>
                <td className="px-5 py-3 font-mono">{row.mean_residual}</td>
                <td className="px-5 py-3 font-mono">{row.std_residual}</td>
                <td className="px-5 py-3 font-mono">{row.median_abs_error}</td>
                <td className="px-5 py-3 font-mono">{row.p90_abs_error}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
