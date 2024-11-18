import React from 'react';
import './App.css';
import ImageGenerator from './ImageGenerator';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <div className="Header-content">
          <div className="Logo-container">
            <svg xmlns="http://www.w3.org/2000/svg" width="50" height="50" viewBox="0 0 100 100">
              {/* Circle background */}
              <circle cx="50" cy="50" r="45" fill="#61dafb" />
              {/* Text */}
              <text x="50" y="55" fontSize="16" textAnchor="middle" fill="#fff" fontWeight="bold">T2FGAN</text>
            </svg>
            <h1 className="App-title">T2FGAN Image Generator</h1>
          </div>
          <div className="Content-wrapper">
            <ImageGenerator />
          </div>
        </div>
      </header>
    </div>
  );
}

export default App;
