import { useState } from 'react';
import './App.css';
import Chatbot from './Chatbot';
import Dashboard from './Dashboard';

function App() {
  // State for URL input
  const [url, setUrl] = useState('');
  // State for the uploaded image file
  const [selectedFile, setSelectedFile] = useState(null);

  // State for handling results, loading, and errors
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Handler for file input change
  const handleFileChange = (event) => {
    // When a file is chosen, clear the URL input and result
    setUrl('');
    setResult(null);
    setSelectedFile(event.target.files[0]);
  };

  const handleUrlSubmit = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });
      if (!response.ok) throw new Error('Network response was not ok');
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Failed to connect to the URL prediction server.');
    }
  };

  const handleImageSubmit = async () => {
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict-image', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('Network response was not ok');
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Failed to connect to the image prediction server.');
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!url && !selectedFile) {
      setError('Please enter a URL or upload a screenshot.');
      return;
    }

    setIsLoading(true);
    setResult(null);
    setError('');

    if (selectedFile) {
      await handleImageSubmit();
    } else {
      await handleUrlSubmit();
    }

    setIsLoading(false);
  };
  
  const resultClass = result ? (result.prediction === 'Phishing' ? 'phishing' : 'legitimate') : '';

  return (
    <div className="app-layout">
      {/* --- Left Sidebar for Dashboard --- */}
      <div className="sidebar-content left">
        <Dashboard />
      </div>

      {/* --- Main Content Column (Center) --- */}
      <div className="main-content">
        <header>
          <h1>Phish-VQC 🎣</h1>
          <p>A Quantum Machine Learning URL & Screenshot Analyzer</p>
        </header>

        <main>
          <form onSubmit={handleSubmit} className="url-form">
            <input
              type="text"
              value={url}
              onChange={(e) => {
                setUrl(e.target.value);
                setSelectedFile(null); // Clear file if user types a URL
              }}
              placeholder="Enter a URL to analyze"
              aria-label="URL Input"
            />
            <button type="submit" disabled={isLoading || (!url && !selectedFile)}>
              {isLoading ? 'Analyzing...' : 'Check'}
            </button>
          </form>

          <div className="separator">OR</div>

          <form onSubmit={handleSubmit} className="file-form">
            <label htmlFor="file-upload" className="custom-file-upload">
                {selectedFile ? selectedFile.name : "Choose Screenshot"}
            </label>
            <input id="file-upload" type="file" accept="image/*" onChange={handleFileChange} />
          </form>


          {error && <div className="result-card error">{error}</div>}

          {result && (
            <div className={`result-card ${resultClass}`}>
              <h2>Analysis Complete</h2>
              <p>The input is likely:</p>
              <p className="prediction">{result.prediction}</p>
            </div>
          )}
        </main>
      </div>

      {/* --- Right Sidebar for Chatbot --- */}
      <div className="sidebar-content right">
        <Chatbot />
      </div>
    </div>
  );
}

export default App;