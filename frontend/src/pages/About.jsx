import React from 'react';
import { Cpu, Layers, GitBranch, ShieldAlert, Award } from 'lucide-react';

export default function About() {
  return (
    <div className="container" style={{ padding: '3.5rem 1.5rem 5rem', flex: 1, maxWidth: '900px' }}>
      <div style={{ marginBottom: '2.5rem' }}>
        <span className="badge badge-accent" style={{ marginBottom: '0.75rem' }}>System Architecture</span>
        <h1 style={{ fontSize: '2.4rem', marginBottom: '0.75rem' }}>About SteelSight</h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '1.05rem', lineHeight: 1.6 }}>
          SteelSight is an industrial-grade computer vision system engineered to localize, decompose, and quantify surface defects on flat steel strips during manufacturing operations.
        </p>
      </div>

      {/* Model Technical Specifications */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1.25rem' }}>
          <Cpu size={22} color="var(--accent)" />
          <h2 style={{ fontSize: '1.25rem' }}>Deep Learning Architecture Specifications</h2>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1.25rem', marginBottom: '1.5rem' }}>
          <div style={{ background: 'var(--surface-elevated)', padding: '1rem', borderRadius: '6px', border: '1px solid var(--border)' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Network Backbone</div>
            <div style={{ fontWeight: 700, fontSize: '1.05rem', marginTop: '0.25rem' }}>ResNet-18 Encoder</div>
          </div>

          <div style={{ background: 'var(--surface-elevated)', padding: '1rem', borderRadius: '6px', border: '1px solid var(--border)' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Decoder Topology</div>
            <div style={{ fontWeight: 700, fontSize: '1.05rem', marginTop: '0.25rem' }}>U-Net (256, 128, 64, 32, 16)</div>
          </div>

          <div style={{ background: 'var(--surface-elevated)', padding: '1rem', borderRadius: '6px', border: '1px solid var(--border)' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Input Resolution</div>
            <div style={{ fontWeight: 700, fontSize: '1.05rem', marginTop: '0.25rem' }}>256 × 1600 px (RGB)</div>
          </div>

          <div style={{ background: 'var(--surface-elevated)', padding: '1rem', borderRadius: '6px', border: '1px solid var(--border)' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Channel Activation</div>
            <div style={{ fontWeight: 700, fontSize: '1.05rem', marginTop: '0.25rem' }}>Independent Sigmoids (4 classes)</div>
          </div>
        </div>

        <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem', lineHeight: 1.6 }}>
          The model utilizes five stages of residual convolutions with lateral skip connections that route high-frequency boundary information directly from encoder feature hierarchies to expanding decoder blocks. This preserves microscopic defect contours even across continuous strip rolling lines.
        </p>
      </div>

      {/* Dataset & Training Context */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1.25rem' }}>
          <Layers size={22} color="var(--success)" />
          <h2 style={{ fontSize: '1.25rem' }}>Dataset & Training Baseline</h2>
        </div>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem', lineHeight: 1.6, marginBottom: '1rem' }}>
          Model weights were trained on the Severstal Steel Defect Detection benchmark corpus containing thousands of annotated 256×1600 surface photographs. Optimization utilized Binary Cross Entropy with Logits combined with Dice coefficient penalization.
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem' }}>
          <span className="badge badge-accent">Epoch: 18</span>
          <span className="badge badge-accent">Best Validation Loss: 0.01176</span>
          <span className="badge badge-accent">Independent Multi-Label Formulation</span>
        </div>
      </div>

      {/* Operational Limitations */}
      <div className="card" style={{ borderLeft: '3px solid var(--warning)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.75rem' }}>
          <ShieldAlert size={22} color="var(--warning)" />
          <h2 style={{ fontSize: '1.25rem' }}>Operational Constraints & Disclaimers</h2>
        </div>
        <ul style={{ color: 'var(--text-muted)', fontSize: '0.9rem', paddingLeft: '1.25rem', lineHeight: 1.6 }}>
          <li>Inference performance depends on consistent lighting and surface cleanliness comparable to typical rolling mill inspection hoods.</li>
          <li>Extreme camera blur, uncalibrated scaling, or non-steel surfaces can produce spurious activations.</li>
          <li>SteelSight provides automated detection assistance and does not supersede ISO 9001 metallurgical certification standards without human oversight.</li>
        </ul>
      </div>
    </div>
  );
}
