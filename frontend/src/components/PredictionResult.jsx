import React from 'react';
import { Bike, CheckCircle2 } from 'lucide-react';

export default function PredictionResult({ singleResult, allResults }) {
  if (!singleResult && !allResults) {
    return null;
  }

  return (
    <div className="space-y-4">
      {singleResult && (
        <div className="rounded-xl border border-blue-200 bg-gradient-to-r from-blue-50 to-white p-6 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-blue-600">
                <CheckCircle2 className="h-4 w-4" />
                <span>Predicted Hourly Bicycle Demand</span>
              </div>
              <p className="mt-2 text-4xl font-bold text-slate-900">
                {singleResult.predicted_count}{' '}
                <span className="text-base font-medium text-slate-500">rentals / hour</span>
              </p>
              <p className="mt-1 text-xs text-slate-500">
                Exact model output: {singleResult.predicted_demand} &bull; Model:{' '}
                <span className="font-semibold text-slate-700">{singleResult.model}</span>
              </p>
            </div>
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-blue-600 text-white shadow">
              <Bike className="h-7 w-7" />
            </div>
          </div>
        </div>
      )}

      {allResults && (
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <h4 className="mb-3 text-base font-semibold text-slate-900">
            Predictions Across All Four Trained Models
          </h4>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {Object.entries(allResults.predictions).map(([modelName, value]) => (
              <div
                key={modelName}
                className="rounded-lg border border-slate-200 bg-slate-50/70 p-4"
              >
                <p className="text-xs font-medium text-slate-500">{modelName}</p>
                <p className="mt-1 text-2xl font-bold text-slate-900">
                  {Math.round(value)}{' '}
                  <span className="text-xs font-normal text-slate-500">({value})</span>
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
