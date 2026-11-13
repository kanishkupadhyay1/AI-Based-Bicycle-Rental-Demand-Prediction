import React, { useEffect, useState } from 'react';
import { Database, Calendar, Layers, Award } from 'lucide-react';
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
import { fetchDemandAnalysis, fetchModelComparison } from '../services/api';
import MetricCard from '../components/MetricCard';
import ModelComparisonChart from '../components/ModelComparisonChart';
import DemandChart from '../components/DemandChart';
import WeatherAnalysisChart from '../components/WeatherAnalysisChart';

export default function Dashboard() {
  const [comparison, setComparison] = useState(null);
  const [demand, setDemand] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        setLoading(true);
        const [compData, demandData] = await Promise.all([
          fetchModelComparison(),
          fetchDemandAnalysis(),
        ]);
        if (active) {
          setComparison(compData);
          setDemand(demandData);
          setError(null);
        }
      } catch (err) {
        if (active) {
          setError(err.response?.data?.detail || err.message || 'Failed to load dashboard data.');
        }
      } finally {
        if (active) setLoading(false);
      }
    };
    load();
    return () => {
      active = false;
    };
  }, []);

  if (loading) {
    return (
      <div className="rounded-xl border border-blue-100 bg-white p-10 text-center text-sm text-slate-500">
        Loading real UCI Bike Sharing Dataset analytics and model metrics...
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-6 text-sm text-red-700">
        <p className="font-semibold">Unable to load dashboard data</p>
        <p className="mt-1">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Total Hourly Records"
          value={comparison.total_records.toLocaleString()}
          subtitle={`${comparison.date_range.start} to ${comparison.date_range.end}`}
          icon={Database}
        />
        <MetricCard
          title="Training Split (80%)"
          value={comparison.train_records.toLocaleString()}
          subtitle="Chronological earlier observations"
          icon={Calendar}
        />
        <MetricCard
          title="Testing Split (20%)"
          value={comparison.test_records.toLocaleString()}
          subtitle="Held-out chronological evaluation"
          icon={Layers}
        />
        <MetricCard
          title="Best Model (Test RMSE)"
          value={comparison.best_model_by_rmse.replace(' Regressor', '')}
          subtitle={`Lowest MAE: ${comparison.best_model_by_mae.replace(' Regressor', '')}`}
          highlight
          icon={Award}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <ModelComparisonChart metrics={comparison.metrics} metricKey="both" />
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h3 className="mb-1 text-base font-semibold text-slate-900">
            Actual vs Predicted Demand (Last 7 Days of Test Set)
          </h3>
          <p className="mb-4 text-xs text-slate-500">
            Chronological test predictions comparing Actual count, GAM, and Gradient Boosting
          </p>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={comparison.recent_time_series}
                margin={{ top: 10, right: 20, left: 0, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="label" tick={{ fontSize: 10 }} interval={23} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Legend verticalAlign="top" height={32} />
                <Line
                  type="monotone"
                  dataKey="actual_cnt"
                  name="Actual"
                  stroke="#0f172a"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="gam_pred"
                  name="GAM"
                  stroke="#2563eb"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="gradient_boosting_pred"
                  name="Gradient Boosting"
                  stroke="#059669"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <DemandChart data={demand.hourly_demand} />
        <WeatherAnalysisChart weatherData={demand.weather_demand} />
      </div>
    </div>
  );
}
