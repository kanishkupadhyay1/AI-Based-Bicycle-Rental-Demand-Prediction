import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';

export default function WeatherAnalysisChart({ weatherData, title = 'Demand by Weather Situation' }) {
  if (!weatherData || weatherData.length === 0) {
    return null;
  }

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <h3 className="mb-1 text-base font-semibold text-slate-900">{title}</h3>
      <p className="mb-4 text-xs text-slate-500">
        Average hourly bicycle rentals across UCI weather categories
      </p>
      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={weatherData} margin={{ top: 10, right: 20, left: 0, bottom: 15 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="label" tick={{ fontSize: 11 }} interval={0} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip formatter={(val) => [`${val} rentals/hr`, 'Avg Demand']} />
            <Bar dataKey="avg_demand" name="Avg Rentals" fill="#3b82f6" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
