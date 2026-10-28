import React, { useState, useEffect } from 'react';
import { useLocation, Link, useNavigate } from 'react-router-dom';
import { Download, ArrowLeft, Layers, AlertCircle, CheckCircle, Eye, Cpu, BarChart2 } from 'lucide-react';

export default function Results() {
  const location = useLocation();
  const navigate = useNavigate();
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState('overlay'); // 'original' | 'overlay' | 'heatmap' | 'class1' | 'class2' | 'class3' | 'class4'

  useEffect(() => {
    // 1. Check router state
    if (location.state && location.state.result) {
      setResult(location.state.result);
    } else {
      // 2. Check session storage fallback
      const cached = sessionStorage.getItem('steelsight_latest_result');
      if (cached) {
        try {
          setResult(JSON.parse(cached));
        } catch (e) {
          console.warn('Failed parsing cached prediction result', e);
        }
      }
    }
  }, [location]);

  if (!result) {
    return (
      <div className="container" style={{ padding: '6rem 1.5rem', textAlign: 'center', flex: 1 }}>
        <div className="card" style={{ maxWidth: '520px', margin: '0 auto', padding: '3rem 2rem' }}>
          <div style={{
            width: '56px',
            height: '56px',
            borderRadius: '50%',
            background: 'var(--surface-elevated)',
            border: '1px solid var(--border)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 1.25rem',
            color: 'var(--warning)'
          }}>
            <AlertCircle size={28} />
          </div>
          <h2 style={{ fontSize: '1.4rem', marginBottom: '0.75rem' }}>No Active Inspection Data</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem', marginBottom: '1.75rem' }}>
            No prior segmentation inference session was found. Upload a steel surface image in the inspection console to view results.
          </p>
          <Link to="/inspect" className="btn btn-primary" style={{ padding: '0.75rem 1.75rem' }}>
            Initiate New Inspection
          </Link>
        </div>
      </div>
    );
  }

  // Determine current active image Data URI
  const getImageSource = () => {
    switch (activeTab) {
      case 'original':
        return result.original_image;
      case 'overlay':
        return result.overlay_image;
      case 'heatmap':
        return result.heatmap_image;
      case 'class1':
        return result.masks[0];
      case 'class2':
        return result.masks[1];
      case 'class3':
        return result.masks[2];
      case 'class4':
        return result.masks[3];
      default:
        return result.overlay_image;
    }
  };

  const downloadImage = (dataUri, filename) => {
    const a = document.createElement('a');
    a.href = dataUri;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="container" style={{ padding: '2.5rem 1.5rem 4rem', flex: 1 }}>
      {/* Top Bar Navigation */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.75rem', flexWrap: 'wrap', gap: '1rem' }}>
        <Link to="/inspect" className="btn btn-secondary" style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}>
          <ArrowLeft size={16} /> New Inspection
        </Link>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span className="badge badge-accent">
            <Cpu size={14} /> {result.device.toUpperCase()}
          </span>
          <span className="badge" style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border)' }}>
            Threshold: {result.threshold.toFixed(2)}
          </span>
          <span className={`badge ${result.any_defect_detected ? 'badge-danger' : 'badge-success'}`}>
            {result.any_defect_detected ? 'Defects Detected' : 'No Anomalies Above Threshold'}
          </span>
        </div>
      </div>

      {/* Main Visualizer Container */}
      <div className="card" style={{ marginBottom: '2rem', padding: '1.25rem' }}>
        {/* Layer Tabs Header */}
        <div style={{
          display: 'flex',
          gap: '0.5rem',
          overflowX: 'auto',
          paddingBottom: '0.75rem',
          borderBottom: '1px solid var(--border)',
          marginBottom: '1rem'
        }}>
          {[
            { id: 'overlay', label: 'Color Overlay' },
            { id: 'original', label: 'Original Strip' },
            { id: 'heatmap', label: 'Confidence Heatmap' },
            { id: 'class1', label: 'Defect Class 1' },
            { id: 'class2', label: 'Defect Class 2' },
            { id: 'class3', label: 'Defect Class 3' },
            { id: 'class4', label: 'Defect Class 4' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: '0.45rem 0.9rem',
                borderRadius: '6px',
                border: '1px solid',
                borderColor: activeTab === tab.id ? 'var(--accent)' : 'var(--border)',
                background: activeTab === tab.id ? 'var(--surface-elevated)' : 'transparent',
                color: activeTab === tab.id ? 'var(--accent)' : 'var(--text-muted)',
                fontWeight: 600,
                fontSize: '0.84rem',
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                transition: 'all 0.15s ease'
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* High-aspect ratio image canvas */}
        <div style={{
          background: '#04070a',
          borderRadius: '6px',
          border: '1px solid var(--border)',
          overflow: 'hidden',
          position: 'relative',
          marginBottom: '1rem'
        }}>
          <img
            src={getImageSource()}
            alt={activeTab}
            style={{
              width: '100%',
              display: 'block',
              aspectRatio: '1600 / 256',
              objectFit: 'contain',
              background: '#04070a'
            }}
          />
        </div>

        {/* Visualizer Footer Bar */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem', fontSize: '0.85rem' }}>
          <div style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            Source: <span style={{ color: 'var(--text)' }}>{result.filename}</span> ({result.original_width} × {result.original_height} px)
          </div>

          <button
            onClick={() => downloadImage(getImageSource(), `${result.filename}_${activeTab}.png`)}
            className="btn btn-secondary"
            style={{ padding: '0.4rem 0.85rem', fontSize: '0.82rem' }}
          >
            <Download size={15} /> Download View
          </button>
        </div>
      </div>

      {/* Quantitative Analysis Breakdown */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        gap: '1.5rem',
        marginBottom: '2rem'
      }}>
        {result.classes.map((cls, idx) => {
          const colors = ['#ff4154', '#45d49a', '#42b8ff', '#f0b45c'];
          const color = colors[idx];

          return (
            <div key={cls.class_id} className="card" style={{ borderLeft: `3px solid ${color}` }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                <span style={{ fontWeight: 700, fontSize: '1rem', color }}>{cls.name}</span>
                <span className={`badge ${cls.detected ? 'badge-danger' : 'badge-success'}`}>
                  {cls.detected ? 'Present' : 'Not Detected'}
                </span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.88rem' }}>
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>Defect Pixels</div>
                  <div style={{ fontWeight: 600, fontFamily: 'var(--font-mono)', fontSize: '1.05rem' }}>
                    {cls.pixels.toLocaleString()} px
                  </div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>Surface Area %</div>
                  <div style={{ fontWeight: 600, fontFamily: 'var(--font-mono)', fontSize: '1.05rem' }}>
                    {cls.area_percent.toFixed(3)}%
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Disclaimer / Responsible-use note */}
      <div style={{
        background: 'rgba(23, 30, 41, 0.6)',
        border: '1px solid var(--border)',
        borderRadius: '6px',
        padding: '1rem 1.25rem',
        fontSize: '0.85rem',
        color: 'var(--text-muted)'
      }}>
        <strong>Responsible Use Note:</strong> Predictions are generated using a pre-trained deep convolutional neural network for automated quality screening assistance. Critical metallurgical certifications must follow standard non-destructive testing (NDT) verification protocols.
      </div>
    </div>
  );
}
