// App.js
import React from 'react';
import FileUpload from './FileUpload';
import './App.css';

function App() {
  return (
    <div className="App">
      <nav className="navbar">
        <div className="nav-logo">NetVision</div>
        <div className="nav-links">
          <a href="#">Upload Network Diagram</a>
          <a href="#">My Workspace</a>
          <a href="#">User Guide</a>
        </div>
      </nav>
      
      <main className="main-content">
        <div className="left-section">
          <h4>NetVision</h4>
          <h1>Redefining network efficiency through computer vision</h1>
          <p>Blurb about NetVision. Maybe something about how easy it is to use...</p>
          <FileUpload />
        </div>
        <div className="right-section">
          <div className="demo-placeholder">
            video demo or img
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;