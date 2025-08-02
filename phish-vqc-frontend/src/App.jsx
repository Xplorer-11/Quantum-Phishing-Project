import { useState } from 'react';
import './App.css';

function App() {
  const [url, setUrl] = useState('');
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!url) {
      setError('Please enter a URL.');
      return;
    }

    setIsLoading(true);
    setResult(null);
    setError('');

    try {
      // The API server runs on port 8000 by default
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: url }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      if (data.error) {
         setError(data.error);
      } else {
         setResult(data);
      }
      
    } catch (err) {
      setError('Failed to connect to the prediction server. Is it running?');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Determine the result card's class based on the prediction
  const resultClass = result ? (result.prediction === 'Phishing' ? 'phishing' : 'legitimate') : '';


  return (
    <div className="container">
      <header>
        <h1>Phish-VQC 🎣</h1>
        <p>A Quantum Machine Learning URL Analyzer</p>
      </header>

      <main>
        <form onSubmit={handleSubmit} className="url-form">
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="Enter a URL to analyze (e.g., https://google.com)"
            aria-label="URL Input"
          />
          <button type="submit" disabled={isLoading}>
            {isLoading ? 'Analyzing...' : 'Check URL'}
          </button>
        </form>

        {error && <div className="result-card error">{error}</div>}

        {result && (
          <div className={`result-card ${resultClass}`}>
            <h2>Analysis Complete</h2>
            <p>The URL <span>{result.url}</span> is likely:</p>
            <p className="prediction">{result.prediction}</p>
          </div>
        )}
      </main>

      <footer>
        <p>Powered by Qiskit and React</p>
      </footer>
    </div>
  );
}

export default App;