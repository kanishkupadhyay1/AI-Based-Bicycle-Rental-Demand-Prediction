import React, { useEffect, useState } from 'react';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import Prediction from './pages/Prediction';
import ModelAnalysis from './pages/ModelAnalysis';
import GamAnalysis from './pages/GamAnalysis';
import About from './pages/About';
import { fetchHealth } from './services/api';

export default function App() {
  const [activePage, setActivePage] = useState('dashboard');
  const [apiHealthy, setApiHealthy] = useState(false);

  useEffect(() => {
    fetchHealth()
      .then((res) => setApiHealthy(res?.status === 'healthy' && res?.models_loaded))
      .catch(() => setApiHealthy(false));
  }, []);

  const renderPage = () => {
    switch (activePage) {
      case 'predict':
        return <Prediction />;
      case 'models':
        return <ModelAnalysis />;
      case 'gam':
        return <GamAnalysis />;
      case 'about':
        return <About />;
      case 'dashboard':
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="min-h-screen">
      <Navbar activePage={activePage} onNavigate={setActivePage} apiHealthy={apiHealthy} />
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6">{renderPage()}</main>
    </div>
  );
}
