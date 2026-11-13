import React from 'react';
import { Bike, BarChart3, Sliders, Activity, Info } from 'lucide-react';

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Overview', icon: BarChart3 },
  { id: 'predict', label: 'Predict Demand', icon: Sliders },
  { id: 'models', label: 'Model Analysis', icon: Activity },
  { id: 'gam', label: 'GAM Smooth Effects', icon: Bike },
  { id: 'about', label: 'About', icon: Info },
];

export default function Navbar({ activePage, onNavigate, apiHealthy }) {
  return (
    <header className="sticky top-0 z-30 border-b border-blue-100 bg-white/95 backdrop-blur">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-600 text-white shadow-sm">
            <Bike className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-base font-bold text-slate-900 sm:text-lg">
              AI-Based Bicycle Rental Demand Prediction
            </h1>
            <p className="text-xs text-slate-500">
              Generalized Additive Models (GAM) &bull; UCI Bike Sharing Dataset
            </p>
          </div>
        </div>

        <nav className="flex flex-wrap items-center gap-1 sm:gap-2">
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => {
            const isActive = activePage === id;
            return (
              <button
                key={id}
                onClick={() => onNavigate(id)}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-slate-600 hover:bg-blue-50 hover:text-blue-700'
                }`}
              >
                <Icon className="h-4 w-4" />
                <span>{label}</span>
              </button>
            );
          })}
          <span
            className={`ml-2 inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${
              apiHealthy
                ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                : 'bg-amber-50 text-amber-700 border border-amber-200'
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                apiHealthy ? 'bg-emerald-500' : 'bg-amber-500'
              }`}
            />
            {apiHealthy ? 'API Connected' : 'Connecting...'}
          </span>
        </nav>
      </div>
    </header>
  );
}
