import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';

const shortName = (name) =>
  name
    .replace('Generalized Additive Model (GAM)', 'GAM')
    .replace(' Regressor', '')
    .replace(' Regression', ' Reg.');

export default function ModelComparisonChart({ metrics, metricKey = 'both' }) {
  if (!metrics || metrics.length === 0) {
    return null;
  }

  const chartData = metrics.map((m) => ({
    ...m,
    shortModel: shortName(m.model),
  }));

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <h3 className="mb-1 text-base font-semibold text-slate-900">
        {metricKey === 'deviance_explained'
          ? 'Poisson Deviance Explained (Higher is Better)'
          : 'Model Error Comparison: MAE & RMSE (Lower is Better)'}
      </h3>
      <p className="mb-4 text-xs text-slate-500">
        Evaluated on the held-out chronological 20% test set
      </p>
      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="shortModel" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip />
            <Legend />
            {metricKey === 'deviance_explained' ? (
              <Bar
                dataKey="deviance_explained"
                name="Deviance Explained"
                fill="#0284c7"
                radius={[6, 6, 0, 0]}
              />
            ) : (
              <>
                <Bar dataKey="mae" name="MAE" fill="#2563eb" radius={[6, 6, 0, 0]} />
                <Bar dataKey="rmse" name="RMSE" fill="#60a5fa" radius={[6, 6, 0, 0]} />
              </>
            )}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
