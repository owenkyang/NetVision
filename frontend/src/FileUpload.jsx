// FileUpload.jsx
import React, { useState } from 'react';
import './FileUpload.css';

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(''); // To display the upload status or result

  const handleFileInput = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setUploadStatus(''); // Clear any previous status
    }
  };

  const handleFileUpload = async () => {
    if (!file) {
      setUploadStatus('Please select a file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setUploadStatus('Uploading...'); // Indicate upload in progress

      const response = await fetch('http://198.202.102.92:5000/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = 'results.csv';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        setUploadStatus('Upload successful! File downloaded.');
      } else {
        setUploadStatus('Upload failed. Please try again.');
      }
    } catch (error) {
      setUploadStatus(`Error: ${error.message}`);
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
          <button onClick={handleFileUpload} className="upload-button">
            Submit
          </button>
        </div>
      )}
      {uploadStatus && <p className="upload-status">{uploadStatus}</p>}
    </div>
  );
};

export default FileUpload;
