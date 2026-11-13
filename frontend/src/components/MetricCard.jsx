import React from 'react';

export default function MetricCard({ title, value, subtitle, highlight = false, icon: Icon }) {
  return (
    <div
      className={`rounded-xl border p-5 shadow-sm transition-shadow hover:shadow-md ${
        highlight
          ? 'border-blue-300 bg-gradient-to-br from-blue-50 to-white'
          : 'border-slate-200 bg-white'
      }`}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
            {title}
          </p>
          <p className="mt-2 text-2xl font-bold text-slate-900">{value}</p>
          {subtitle && <p className="mt-1 text-xs text-slate-500">{subtitle}</p>}
        </div>
        {Icon && (
          <div className="rounded-lg bg-blue-50 p-2.5 text-blue-600">
            <Icon className="h-5 w-5" />
          </div>
        )}
      </div>
    </div>
  );
}
