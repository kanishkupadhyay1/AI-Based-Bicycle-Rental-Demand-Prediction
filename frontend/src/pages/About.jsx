import React from 'react';

export default function About() {
  return (
    <div className="space-y-6 rounded-xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">
          AI-Based Bicycle Rental Demand Prediction Using Generalized Additive Models (GAM)
        </h2>
        <p className="mt-2 text-sm text-slate-600">
          This project models nonlinear bicycle rental demand relationships with weather and
          temporal variables using a Poisson Generalized Additive Model (GAM) and benchmarks its
          performance against Linear Regression, Random Forest, and Gradient Boosting Regressors.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <div className="rounded-lg border border-slate-100 bg-slate-50 p-5">
          <h3 className="text-base font-semibold text-slate-900">Dataset Source</h3>
          <p className="mt-2 text-sm text-slate-600">
            Official <strong>UCI Machine Learning Repository: Bike Sharing Dataset</strong>{' '}
            (hourly records from Capital Bikeshare system, Washington D.C., 2011–2012).
          </p>
          <a
            href="https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
            target="_blank"
            rel="noreferrer"
            className="mt-2 inline-block text-sm font-medium text-blue-600 hover:underline"
          >
            https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
          </a>
        </div>

        <div className="rounded-lg border border-slate-100 bg-slate-50 p-5">
          <h3 className="text-base font-semibold text-slate-900">Methodology &amp; Split</h3>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-600">
            <li>Chronological 80% train / 20% test split (no temporal data leakage).</li>
            <li>Excluded identifier (instant) and target components (casual, registered).</li>
            <li>Smooth cubic spline terms for hr, temp, atemp, hum, windspeed, mnth.</li>
            <li>Factor terms for season, yr, holiday, weekday, workingday, weathersit.</li>
          </ul>
        </div>

        <div className="rounded-lg border border-slate-100 bg-slate-50 p-5">
          <h3 className="text-base font-semibold text-slate-900">Evaluation Metrics</h3>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-600">
            <li><strong>MAE</strong>: Mean Absolute Error in rental counts per hour.</li>
            <li><strong>RMSE</strong>: Root Mean Squared Error penalizing large peak errors.</li>
            <li>
              <strong>Poisson Deviance Explained</strong>: Proportion of null Poisson deviance
              explained on the held-out test set.
            </li>
          </ul>
        </div>

        <div className="rounded-lg border border-slate-100 bg-slate-50 p-5">
          <h3 className="text-base font-semibold text-slate-900">Technology Stack</h3>
          <p className="mt-2 text-sm text-slate-600">
            Python 3.11, PyGAM, Scikit-learn, Pandas, NumPy, FastAPI, Uvicorn, React, Vite,
            Tailwind CSS, Recharts, Axios, and Pytest.
          </p>
        </div>
      </div>
    </div>
  );
}
