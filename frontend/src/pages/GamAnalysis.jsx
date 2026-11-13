import React, { useEffect, useState } from 'react';
import { fetchGamEffects } from '../services/api';
import PartialEffectChart from '../components/PartialEffectChart';

export default function GamAnalysis() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    fetchGamEffects()
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
        Extracting PoissonGAM smooth partial dependence curves...
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
          PoissonGAM Smooth Partial Dependence Effects
        </h2>
        <p className="mt-1 text-sm text-slate-700">{data.interpretation}</p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {data.effects.map((curve) => (
          <PartialEffectChart key={curve.feature} curve={curve} />
        ))}
      </div>
    </div>
  );
}
