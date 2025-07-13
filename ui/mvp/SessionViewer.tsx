// Example in a React component in ui/dashboard/src/components/AnalysisUploader.tsx

import React, { useState } from 'react';

function AnalysisUploader() {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    setError('');
    setAnalysisResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // 1. Upload file, create session, and get session_id back
      const uploadResponse = await fetch('/api/analyze_text_session', {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error('Failed to upload and process file.');
      }

      const { session_id } = await uploadResponse.json();

      // 2. Use the session_id to fetch the analysis results
      const resultsResponse = await fetch(`/api/analysis/${session_id}`);
      if (!resultsResponse.ok) {
        throw new Error('Failed to fetch analysis results.');
      }
      
      const results = await resultsResponse.json();
      setAnalysisResult(results);

    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <h1>Upload Therapy Transcript (.txt)</h1>
      <input type="file" accept=".txt" onChange={handleFileChange} disabled={isLoading} />
      
      {isLoading && <p>Analyzing... please wait.</p>}
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}
      
      {analysisResult && (
        <div>
          <h2>Analysis Results</h2>
          <pre>{JSON.stringify(analysisResult, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default AnalysisUploader;