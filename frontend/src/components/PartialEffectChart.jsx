import React from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts';

export default function PartialEffectChart({ curve }) {
  if (!curve || !curve.points || curve.points.length === 0) {
    return null;
  }

  const data = curve.points.map((pt) => ({
    x: pt.x,
    effect: pt.effect,
    band: [pt.ci_lower, pt.ci_upper],
  }));

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <h3 className="mb-1 text-base font-semibold text-slate-900">{curve.label}</h3>
      <p className="mb-3 text-xs text-slate-500">
        Zero-centered smooth partial effect (log-link) with 95% confidence interval
      </p>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 15, left: -10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="x" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip
              formatter={(value, name) => {
                if (Array.isArray(value)) {
                  return [`[${value[0]}, ${value[1]}]`, '95% CI'];
                }
                return [value, name];
              }}
            />
            <ReferenceLine y={0} stroke="#64748b" strokeDasharray="4 4" />
            <Area
              type="monotone"
              dataKey="band"
              stroke="none"
              fill="#93c5fd"
              fillOpacity={0.35}
              name="95% CI"
            />
            <Line
              type="monotone"
              dataKey="effect"
              stroke="#1d4ed8"
              strokeWidth={2.5}
              dot={false}
              name="Partial Effect"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
