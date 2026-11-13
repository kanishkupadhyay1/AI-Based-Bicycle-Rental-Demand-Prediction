import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';

export default function DemandChart({ data, title = 'Hourly Rental Demand Pattern' }) {
  if (!data || data.length === 0) {
    return null;
  }

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <h3 className="mb-1 text-base font-semibold text-slate-900">{title}</h3>
      <p className="mb-4 text-xs text-slate-500">
        Mean hourly rental count (cnt) comparing working days vs weekends/holidays
      </p>
      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="hr" tick={{ fontSize: 12 }} label={{ value: 'Hour (0-23)', position: 'insideBottom', offset: -2, fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip />
            <Legend verticalAlign="top" height={32} />
            <Line
              type="monotone"
              dataKey="working_day"
              name="Working Day"
              stroke="#2563eb"
              strokeWidth={2.5}
              dot={{ r: 3 }}
            />
            <Line
              type="monotone"
              dataKey="non_working_day"
              name="Weekend / Holiday"
              stroke="#0ea5e9"
              strokeWidth={2.5}
              dot={{ r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
