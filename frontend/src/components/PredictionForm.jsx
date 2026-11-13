import React, { useState, useEffect } from 'react';
import { Play, Layers } from 'lucide-react';

const MODEL_OPTIONS = [
  { key: 'gam', label: 'Generalized Additive Model (GAM)' },
  { key: 'linear_regression', label: 'Linear Regression' },
  { key: 'random_forest', label: 'Random Forest Regressor' },
  { key: 'gradient_boosting', label: 'Gradient Boosting Regressor' },
];

const SEASON_LABELS = { 1: '1 - Spring', 2: '2 - Summer', 3: '3 - Fall', 4: '4 - Winter' };
const YEAR_LABELS = { 0: '0 - 2011', 1: '1 - 2012' };
const WEEKDAY_LABELS = {
  0: '0 - Sunday',
  1: '1 - Monday',
  2: '2 - Tuesday',
  3: '3 - Wednesday',
  4: '4 - Thursday',
  5: '5 - Friday',
  6: '6 - Saturday',
};
const WEATHER_LABELS = {
  1: '1 - Clear / Partly Cloudy',
  2: '2 - Mist / Cloudy',
  3: '3 - Light Snow / Rain',
  4: '4 - Heavy Rain / Snow',
};

export default function PredictionForm({ featureInfo, onPredictSingle, onPredictAll, loading }) {
  const [formValues, setFormValues] = useState(null);
  const [selectedModel, setSelectedModel] = useState('gam');

  useEffect(() => {
    if (featureInfo && featureInfo.ranges && !formValues) {
      const defaults = {};
      Object.entries(featureInfo.ranges).forEach(([key, meta]) => {
        defaults[key] = meta.default;
      });
      setFormValues(defaults);
    }
  }, [featureInfo, formValues]);

  if (!formValues || !featureInfo?.ranges) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-500">
        Loading dataset-derived feature bounds...
      </div>
    );
  }

  const ranges = featureInfo.ranges;

  const handleChange = (field, value, isInt = false) => {
    setFormValues((prev) => ({
      ...prev,
      [field]: isInt ? parseInt(value, 10) : parseFloat(value),
    }));
  };

  const handleSubmitSingle = (e) => {
    e.preventDefault();
    onPredictSingle({ ...formValues, model: selectedModel });
  };

  const handleCompareAll = () => {
    onPredictAll({ ...formValues });
  };

  return (
    <form
      onSubmit={handleSubmitSingle}
      className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm"
    >
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div>
          <h3 className="text-lg font-semibold text-slate-900">
            Weather &amp; Temporal Input Parameters
          </h3>
          <p className="text-xs text-slate-500">
            Bounds and defaults are derived directly from the UCI Bike Sharing training dataset.
          </p>
        </div>
        <div className="w-full sm:w-72">
          <label className="mb-1 block text-xs font-medium text-slate-600">
            Primary Prediction Model
          </label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 focus:border-blue-500 focus:outline-none"
          >
            {MODEL_OPTIONS.map((opt) => (
              <option key={opt.key} value={opt.key}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <div>
          <label className="mb-1 block text-xs font-medium text-slate-700">
            Hour of Day ({ranges.hr.min}–{ranges.hr.max})
          </label>
          <input
            type="number"
            min={ranges.hr.min}
            max={ranges.hr.max}
            step={1}
            value={formValues.hr}
            onChange={(e) => handleChange('hr', e.target.value, true)}
            className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
          />
        </div>

        <div>
          <label className="mb-1 block text-xs font-medium text-slate-700">
            Month ({ranges.mnth.min}–{ranges.mnth.max})
          </label>
          <input
            type="number"
            min={ranges.mnth.min}
            max={ranges.mnth.max}
            step={1}
            value={formValues.mnth}
            onChange={(e) => handleChange('mnth', e.target.value, true)}
            className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
          />
        </div>

        <div>
          <label className="mb-1 block text-xs font-medium text-slate-700">Season</label>
          <select
            value={formValues.season}
            onChange={(e) => handleChange('season', e.target.value, true)}
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
          >
            {(ranges.season.unique_values || [1, 2, 3, 4]).map((val) => (
              <option key={val} value={val}>
                {SEASON_LABELS[val] || val}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="mb-1 block text-xs font-medium text-slate-700">Year</label>
          <select
            value={formValues.yr}
            onChange={(e) => handleChange('yr', e.target.value, true)}
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
          >
            {(ranges.yr.unique_values || [0, 1]).map((val) => (
              <option key={val} value={val}>
                {YEAR_LABELS[val] || val}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="mb-1 block text-xs font-medium text-slate-700">Weekday</label>
          <select
            value={formValues.weekday}
            onChange={(e) => handleChange('weekday', e.target.value, true)}
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
          >
            {(ranges.weekday.unique_values || [0, 1, 2, 3, 4, 5, 6]).map((val) => (
              <option key={val} value={val}>
                {WEEKDAY_LABELS[val] || val}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="mb-1 block text-xs font-medium text-slate-700">Weather Situation</label>
          <select
            value={formValues.weathersit}
            onChange={(e) => handleChange('weathersit', e.target.value, true)}
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
          >
            {[1, 2, 3, 4].map((val) => (
              <option key={val} value={val}>
                {WEATHER_LABELS[val]}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="mb-1 block text-xs font-medium text-slate-700">Working Day</label>
          <select
            value={formValues.workingday}
            onChange={(e) => handleChange('workingday', e.target.value, true)}
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
          >
            <option value={1}>1 - Yes (Working Day)</option>
            <option value={0}>0 - No (Weekend / Holiday)</option>
          </select>
        </div>

        <div>
          <label className="mb-1 block text-xs font-medium text-slate-700">Holiday</label>
          <select
            value={formValues.holiday}
            onChange={(e) => handleChange('holiday', e.target.value, true)}
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
          >
            <option value={0}>0 - No</option>
            <option value={1}>1 - Yes</option>
          </select>
        </div>

        <div>
          <label className="mb-1 flex justify-between text-xs font-medium text-slate-700">
            <span>Normalized Temp</span>
            <span className="font-mono text-blue-600">{formValues.temp}</span>
          </label>
          <input
            type="range"
            min={ranges.temp.min}
            max={ranges.temp.max}
            step={0.01}
            value={formValues.temp}
            onChange={(e) => handleChange('temp', e.target.value)}
            className="w-full accent-blue-600"
          />
          <div className="flex justify-between text-[10px] text-slate-400">
            <span>{ranges.temp.min}</span>
            <span>{ranges.temp.max}</span>
          </div>
        </div>

        <div>
          <label className="mb-1 flex justify-between text-xs font-medium text-slate-700">
            <span>Feeling Temp (atemp)</span>
            <span className="font-mono text-blue-600">{formValues.atemp}</span>
          </label>
          <input
            type="range"
            min={ranges.atemp.min}
            max={ranges.atemp.max}
            step={0.01}
            value={formValues.atemp}
            onChange={(e) => handleChange('atemp', e.target.value)}
            className="w-full accent-blue-600"
          />
          <div className="flex justify-between text-[10px] text-slate-400">
            <span>{ranges.atemp.min}</span>
            <span>{ranges.atemp.max}</span>
          </div>
        </div>

        <div>
          <label className="mb-1 flex justify-between text-xs font-medium text-slate-700">
            <span>Normalized Humidity</span>
            <span className="font-mono text-blue-600">{formValues.hum}</span>
          </label>
          <input
            type="range"
            min={ranges.hum.min}
            max={ranges.hum.max}
            step={0.01}
            value={formValues.hum}
            onChange={(e) => handleChange('hum', e.target.value)}
            className="w-full accent-blue-600"
          />
          <div className="flex justify-between text-[10px] text-slate-400">
            <span>{ranges.hum.min}</span>
            <span>{ranges.hum.max}</span>
          </div>
        </div>

        <div>
          <label className="mb-1 flex justify-between text-xs font-medium text-slate-700">
            <span>Normalized Windspeed</span>
            <span className="font-mono text-blue-600">{formValues.windspeed}</span>
          </label>
          <input
            type="range"
            min={ranges.windspeed.min}
            max={ranges.windspeed.max}
            step={0.01}
            value={formValues.windspeed}
            onChange={(e) => handleChange('windspeed', e.target.value)}
            className="w-full accent-blue-600"
          />
          <div className="flex justify-between text-[10px] text-slate-400">
            <span>{ranges.windspeed.min}</span>
            <span>{ranges.windspeed.max}</span>
          </div>
        </div>
      </div>

      <div className="mt-6 flex flex-wrap gap-3">
        <button
          type="submit"
          disabled={loading}
          className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-blue-700 disabled:opacity-50"
        >
          <Play className="h-4 w-4" />
          {loading ? 'Predicting...' : 'Predict Demand'}
        </button>

        <button
          type="button"
          disabled={loading}
          onClick={handleCompareAll}
          className="inline-flex items-center gap-2 rounded-lg border border-blue-200 bg-blue-50 px-5 py-2.5 text-sm font-semibold text-blue-700 hover:bg-blue-100 disabled:opacity-50"
        >
          <Layers className="h-4 w-4" />
          Compare All 4 Models
        </button>
      </div>
    </form>
  );
}
