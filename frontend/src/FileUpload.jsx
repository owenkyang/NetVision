// FileUpload.jsx
import React, { useState } from 'react';
import './FileUpload.css';

const FileUpload = () => {
  const [file, setFile] = useState(null);

  const handleFileInput = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  return (
    <div className="upload-container">
      <input
        type="file"
        id="file-input"
        onChange={handleFileInput}
        className="file-input"
        hidden
      />
      <label htmlFor="file-input" className="upload-button">
        Upload File
      </label>
      {file && (
        <div className="file-info">
          <p>Selected file: {file.name}</p>
        </div>
      )}
    </div>
  );
};

export default FileUpload;