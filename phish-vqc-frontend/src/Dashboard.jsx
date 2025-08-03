import React, { useState, useEffect } from 'react';

const Dashboard = () => {
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({ top_phishing_sites: [] });
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch recent history
        const historyRes = await fetch('http://127.0.0.1:8000/history');
        const historyData = await historyRes.json();
        setHistory(historyData);

        // Fetch stats
        const statsRes = await fetch('http://127.0.0.1:8000/stats');
        const statsData = await statsRes.json();
        setStats(statsData);

      } catch (err) {
        setError('Could not load dashboard data. Is the backend server running?');
      }
    };

    fetchData();
  }, []);

  if (error) {
    return <div className="dashboard-container error">{error}</div>;
  }

  return (
    <div className="dashboard-container">
      <h2>Analytics Dashboard</h2>
      <div className="dashboard-grid">
        <div className="dashboard-card">
          <h3>Recent Analyses</h3>
          {history.length > 0 ? (
            <ul>
              {history.map((item, index) => (
                <li key={index} className={item.prediction.toLowerCase()}>
                  <span>{item.url}</span>
                  <strong>{item.prediction}</strong>
                </li>
              ))}
            </ul>
          ) : <p>No history yet. Analyze some URLs to see data here.</p>}
        </div>
        <div className="dashboard-card">
          <h3>Top Reported Phishing Sites</h3>
          {stats.top_phishing_sites.length > 0 ? (
            <ul>
              {stats.top_phishing_sites.map((item, index) => (
                <li key={index} className="phishing">
                  <span>{item.url}</span>
                  <strong>Reported {item.count} times</strong>
                </li>
              ))}
            </ul>
          ) : <p>No phishing sites reported yet.</p>}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;