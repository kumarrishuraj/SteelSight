import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, Layers, Target, ShieldCheck, Activity, Cpu, Sparkles } from 'lucide-react';

export default function Home() {
  return (
    <div style={{ paddingBottom: '4rem' }}>
      {/* Hero Section */}
      <section style={{
        padding: '5rem 0 4rem',
        borderBottom: '1px solid var(--border)',
        position: 'relative'
      }}>
        <div className="container">
          <div style={{ maxWidth: '800px', margin: '0 auto', textAlign: 'center' }}>
            <div style={{ display: 'inline-flex', marginBottom: '1.25rem' }}>
              <span className="badge badge-accent" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Sparkles size={14} /> Production Deep Learning Pipeline
              </span>
            </div>

            <h1 style={{
              fontSize: 'clamp(2.2rem, 5vw, 3.6rem)',
              fontWeight: 800,
              letterSpacing: '-0.03em',
              marginBottom: '1.25rem',
              color: 'var(--text)'
            }}>
              See defects before they become <span style={{ color: 'var(--accent)', textShadow: '0 0 24px rgba(66, 184, 255, 0.4)' }}>problems.</span>
            </h1>

            <p style={{
              fontSize: '1.15rem',
              color: 'var(--text-muted)',
              marginBottom: '2.5rem',
              lineHeight: 1.6
            }}>
              High-resolution pixel-level semantic segmentation for hot and cold rolled sheet steel. Powered by a custom ResNet-18 U-Net encoder-decoder trained for multi-class metallurgical defect isolation.
            </p>

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', justifyContent: 'center' }}>
              <Link to="/inspect" className="btn btn-primary" style={{ padding: '0.85rem 2rem', fontSize: '1.05rem' }}>
                Start Inspection <ArrowRight size={18} />
              </Link>
              <Link to="/about" className="btn btn-secondary" style={{ padding: '0.85rem 1.75rem', fontSize: '1.05rem' }}>
                Explore the System
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Feature Capabilities Grid */}
      <section style={{ padding: '4.5rem 0' }}>
        <div className="container">
          <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
            <span className="badge badge-accent" style={{ marginBottom: '0.75rem' }}>Core Capabilities</span>
            <h2 style={{ fontSize: '2rem', fontWeight: 700 }}>Engineered for Industrial Inspection</h2>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '1.5rem'
          }}>
            <div className="card">
              <div style={{
                width: '44px',
                height: '44px',
                borderRadius: '8px',
                background: 'rgba(66, 184, 255, 0.1)',
                border: '1px solid rgba(66, 184, 255, 0.25)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '1.25rem',
                color: 'var(--accent)'
              }}>
                <Layers size={22} />
              </div>
              <h3 style={{ fontSize: '1.2rem', marginBottom: '0.6rem' }}>4-Channel Defect Decomposition</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem' }}>
                Deconstructs surface anomalies into four distinct defect categories simultaneously via sigmoid-activated prediction heads.
              </p>
            </div>

            <div className="card">
              <div style={{
                width: '44px',
                height: '44px',
                borderRadius: '8px',
                background: 'rgba(69, 212, 154, 0.1)',
                border: '1px solid rgba(69, 212, 154, 0.25)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '1.25rem',
                color: 'var(--success)'
              }}>
                <Target size={22} />
              </div>
              <h3 style={{ fontSize: '1.2rem', marginBottom: '0.6rem' }}>Fine Spatial Precision</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem' }}>
                Processes extreme aspect-ratio continuous steel strip images (1600×256 pixels) without loss of longitudinal edge fidelity.
              </p>
            </div>

            <div className="card">
              <div style={{
                width: '44px',
                height: '44px',
                borderRadius: '8px',
                background: 'rgba(240, 180, 92, 0.1)',
                border: '1px solid rgba(240, 180, 92, 0.25)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '1.25rem',
                color: 'var(--warning)'
              }}>
                <Activity size={22} />
              </div>
              <h3 style={{ fontSize: '1.2rem', marginBottom: '0.6rem' }}>Confidence Heatmaps</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem' }}>
                Generates perceptual density gradient maps revealing subtle micro-fissures and marginal surface irregularities before total failure.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Industrial Inspection Workflow */}
      <section style={{
        padding: '4rem 0',
        background: 'var(--surface)',
        borderTop: '1px solid var(--border)',
        borderBottom: '1px solid var(--border)'
      }}>
        <div className="container">
          <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
            <span className="badge badge-accent" style={{ marginBottom: '0.75rem' }}>Inspection Pipeline</span>
            <h2 style={{ fontSize: '2rem', fontWeight: 700 }}>How SteelSight Operates</h2>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
            gap: '2rem',
            position: 'relative'
          }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{
                width: '56px',
                height: '56px',
                borderRadius: '50%',
                background: 'var(--surface-elevated)',
                border: '2px solid var(--accent)',
                margin: '0 auto 1.25rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '1.2rem',
                fontWeight: 700,
                color: 'var(--accent)'
              }}>01</div>
              <h4 style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>Image Ingestion</h4>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                Upload high-definition surface captures directly from shop floor cameras or line scanners.
              </p>
            </div>

            <div style={{ textAlign: 'center' }}>
              <div style={{
                width: '56px',
                height: '56px',
                borderRadius: '50%',
                background: 'var(--surface-elevated)',
                border: '2px solid var(--accent)',
                margin: '0 auto 1.25rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '1.2rem',
                fontWeight: 700,
                color: 'var(--accent)'
              }}>02</div>
              <h4 style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>U-Net Forward Pass</h4>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                ResNet-18 features are extracted across 5 stages and reconstructed via skip connections.
              </p>
            </div>

            <div style={{ textAlign: 'center' }}>
              <div style={{
                width: '56px',
                height: '56px',
                borderRadius: '50%',
                background: 'var(--surface-elevated)',
                border: '2px solid var(--accent)',
                margin: '0 auto 1.25rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '1.2rem',
                fontWeight: 700,
                color: 'var(--accent)'
              }}>03</div>
              <h4 style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>Quantitative Analysis</h4>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                Instant breakdown of affected pixel area, defect overlays, and channel-by-channel mask exports.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Call To Action Banner */}
      <section style={{ padding: '4.5rem 0' }}>
        <div className="container">
          <div style={{
            background: 'linear-gradient(135deg, var(--surface-elevated) 0%, var(--surface) 100%)',
            border: '1px solid var(--border)',
            borderRadius: '12px',
            padding: '3rem 2rem',
            textAlign: 'center',
            position: 'relative',
            overflow: 'hidden'
          }}>
            <h2 style={{ fontSize: '2.2rem', marginBottom: '1rem' }}>Initiate Steel Surface Inspection</h2>
            <p style={{ color: 'var(--text-muted)', maxWidth: '600px', margin: '0 auto 2rem', fontSize: '1.05rem' }}>
              Test your industrial production captures against the loaded model checkpoint to receive instant segmentation analytics.
            </p>
            <Link to="/inspect" className="btn btn-primary" style={{ padding: '0.85rem 2.2rem', fontSize: '1.05rem' }}>
              Open Inspection Console <ArrowRight size={18} />
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
