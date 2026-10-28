import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { UploadCloud, Image as ImageIcon, Sliders, AlertTriangle, CheckCircle2, RefreshCw, X } from 'lucide-react';
import { runInference, fetchModelStatus } from '../api/client';

export default function Inspect() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [imageMeta, setImageMeta] = useState(null);
  const [threshold, setThreshold] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    fetchModelStatus()
      .then(setModelStatus)
      .catch((err) => console.warn('Could not fetch model status', err));
  }, []);

  const handleFileChange = (selectedFile) => {
    setError(null);
    if (!selectedFile) return;

    // Validate size (15MB)
    if (selectedFile.size > 15 * 1024 * 1024) {
      setError('File exceeds 15 MB limit. Please select a smaller image.');
      return;
    }

    // Validate type
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    if (!validTypes.includes(selectedFile.type) && !selectedFile.name.match(/\.(jpg|jpeg|png|webp|bmp)$/i)) {
      setError('Unsupported file type. Please upload a JPG, PNG, WEBP, or BMP file.');
      return;
    }

    setFile(selectedFile);
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);

    // Read dimensions
    const img = new Image();
    img.onload = () => {
      setImageMeta({
        width: img.naturalWidth,
        height: img.naturalHeight,
        sizeKb: Math.round(selectedFile.size / 1024)
      });
    };
    img.src = url;
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const clearSelection = () => {
    setFile(null);
    setPreviewUrl(null);
    setImageMeta(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select or drop an image first.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await runInference(file, threshold);
      // Persist in sessionStorage for page refreshes & results access
      sessionStorage.setItem('steelsight_latest_result', JSON.stringify(result));
      navigate('/results', { state: { result } });
    } catch (err) {
      setError(err.message || 'Inspection failed. Please verify API service is active.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container" style={{ padding: '3rem 1.5rem', flex: 1 }}>
      <div style={{ maxWidth: '840px', margin: '0 auto' }}>
        
        {/* Header */}
        <div style={{ marginBottom: '2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.5rem' }}>
            <span className="badge badge-accent">Inspection Console</span>
            {modelStatus && (
              <span className={`badge ${modelStatus.loaded ? 'badge-success' : 'badge-danger'}`}>
                {modelStatus.loaded ? `Model Ready (${modelStatus.device.toUpperCase()})` : 'Model Offline'}
              </span>
            )}
          </div>
          <h1 style={{ fontSize: '2.2rem', marginBottom: '0.5rem' }}>Steel Surface Inspection</h1>
          <p style={{ color: 'var(--text-muted)' }}>
            Upload high-resolution continuous strip images to perform real-time pixel-level segmentation across 4 defect channels.
          </p>
        </div>

        {/* Error Notification */}
        {error && (
          <div style={{
            background: 'rgba(255, 101, 119, 0.1)',
            border: '1px solid var(--danger)',
            borderRadius: '6px',
            padding: '1rem',
            marginBottom: '1.5rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            color: 'var(--danger)'
          }}>
            <AlertTriangle size={20} />
            <div style={{ fontSize: '0.92rem' }}>{error}</div>
          </div>
        )}

        <form onSubmit={handleSubmit}>
          {/* Upload Dropzone */}
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            style={{
              background: isDragging ? 'var(--surface-hover)' : 'var(--surface)',
              border: `2px dashed ${isDragging ? 'var(--accent)' : 'var(--border)'}`,
              borderRadius: '8px',
              padding: '2.5rem 1.5rem',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              marginBottom: '1.5rem'
            }}
            onClick={() => fileInputRef.current && fileInputRef.current.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => handleFileChange(e.target.files[0])}
              accept=".jpg,.jpeg,.png,.webp,.bmp"
              style={{ display: 'none' }}
            />

            <div style={{
              width: '54px',
              height: '54px',
              borderRadius: '50%',
              background: 'var(--surface-elevated)',
              border: '1px solid var(--border)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 1rem',
              color: 'var(--accent)'
            }}>
              <UploadCloud size={28} />
            </div>

            <h3 style={{ fontSize: '1.15rem', marginBottom: '0.4rem' }}>
              Drag and drop surface image here
            </h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', marginBottom: '1rem' }}>
              Supports JPG, PNG, WEBP, BMP up to 15 MB
            </p>
            <button
              type="button"
              className="btn btn-secondary"
              style={{ fontSize: '0.85rem', padding: '0.4rem 1rem' }}
              onClick={(e) => {
                e.stopPropagation();
                fileInputRef.current && fileInputRef.current.click();
              }}
            >
              Browse Files
            </button>
          </div>

          {/* Selected Image Preview & Metadata */}
          {file && previewUrl && (
            <div className="card" style={{ marginBottom: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <ImageIcon size={18} color="var(--accent)" />
                  <span style={{ fontWeight: 600, fontSize: '0.95rem' }}>{file.name}</span>
                </div>
                <button
                  type="button"
                  onClick={clearSelection}
                  style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.2rem', fontSize: '0.85rem' }}
                >
                  <X size={16} /> Remove
                </button>
              </div>

              {/* Aspect-ratio preserving preview container */}
              <div style={{
                background: '#04070a',
                borderRadius: '6px',
                border: '1px solid var(--border)',
                overflow: 'hidden',
                maxHeight: '200px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '1rem'
              }}>
                <img
                  src={previewUrl}
                  alt="Preview"
                  style={{ maxWidth: '100%', maxHeight: '200px', objectFit: 'contain' }}
                />
              </div>

              {imageMeta && (
                <div style={{ display: 'flex', gap: '1.5rem', fontSize: '0.82rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  <div>Dimensions: <span style={{ color: 'var(--text)' }}>{imageMeta.width} × {imageMeta.height} px</span></div>
                  <div>File Size: <span style={{ color: 'var(--text)' }}>{imageMeta.sizeKb} KB</span></div>
                  <div>Model Target: <span style={{ color: 'var(--accent)' }}>1600 × 256 px</span></div>
                </div>
              )}
            </div>
          )}

          {/* Inference Settings: Threshold slider */}
          <div className="card" style={{ marginBottom: '2rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Sliders size={18} color="var(--accent)" />
                <span style={{ fontWeight: 600, fontSize: '0.95rem' }}>Detection Threshold</span>
              </div>
              <span className="badge badge-accent" style={{ fontFamily: 'var(--font-mono)' }}>
                {threshold.toFixed(2)}
              </span>
            </div>
            
            <input
              type="range"
              min="0.10"
              max="0.90"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              style={{
                width: '100%',
                accentColor: 'var(--accent)',
                cursor: 'pointer'
              }}
            />

            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>
              <span>High Sensitivity (0.10)</span>
              <span>Recommended (0.50)</span>
              <span>High Specificity (0.90)</span>
            </div>
          </div>

          {/* Action Trigger */}
          <button
            type="submit"
            disabled={!file || loading}
            className="btn btn-primary"
            style={{ width: '100%', padding: '1rem', fontSize: '1.05rem' }}
          >
            {loading ? (
              <>
                <RefreshCw size={20} className="spin" style={{ animation: 'spin 1s linear infinite' }} />
                Executing PyTorch Segmentation...
              </>
            ) : (
              'Run Defect Inspection'
            )}
          </button>
        </form>
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
