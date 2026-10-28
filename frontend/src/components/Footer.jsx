import React from 'react';
import { Eye, Shield, Terminal } from 'lucide-react';

export default function Footer() {
  return (
    <footer style={{
      borderTop: '1px solid var(--border)',
      background: 'var(--surface)',
      marginTop: 'auto',
      padding: '3rem 0 2rem'
    }}>
      <div className="container">
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
          gap: '2.5rem',
          marginBottom: '2.5rem'
        }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.8rem' }}>
              <Eye size={20} color="var(--accent)" />
              <span style={{ fontWeight: 800, fontSize: '1.1rem' }}>SteelSight</span>
            </div>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', maxWidth: '320px' }}>
              High-throughput deep learning segmentation for industrial steel manufacturing and automated quality inspection.
            </p>
          </div>

          <div>
            <h4 style={{ fontSize: '0.95rem', marginBottom: '1rem', color: 'var(--text)' }}>System Capabilities</h4>
            <ul style={{ listStyle: 'none', color: 'var(--text-muted)', fontSize: '0.88rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <li>4-Class Defect Segmentation</li>
              <li>Multi-Channel Confidence Heatmaps</li>
              <li>Pixel-level Area Quantification</li>
              <li>Real-time PyTorch CPU/CUDA Inference</li>
            </ul>
          </div>

          <div>
            <h4 style={{ fontSize: '0.95rem', marginBottom: '1rem', color: 'var(--text)' }}>Architecture Stack</h4>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
              <span className="badge badge-accent">FastAPI</span>
              <span className="badge badge-accent">PyTorch 2.13</span>
              <span className="badge badge-accent">ResNet-18 U-Net</span>
              <span className="badge badge-accent">React 18 + Vite</span>
            </div>
          </div>
        </div>

        <div style={{
          borderTop: '1px solid var(--border)',
          paddingTop: '1.5rem',
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: '0.85rem',
          color: 'var(--text-muted)'
        }}>
          <div>© {new Date().getFullYear()} SteelSight System. Designed for precision metallurgical quality assurance.</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Terminal size={14} color="var(--accent)" />
            <span>Industrial CV Inference Node</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
