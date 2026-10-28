import React, { useState } from 'react';
import { Mail, MessageSquare, Send, CheckCircle2, AlertCircle } from 'lucide-react';

export default function Contact() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });
  const [errors, setErrors] = useState({});
  const [submitted, setSubmitted] = useState(false);

  const validate = () => {
    const errs = {};
    if (!formData.name.trim()) errs.name = 'Full name is required.';
    if (!formData.email.trim()) {
      errs.email = 'Email address is required.';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      errs.email = 'Please provide a valid email address.';
    }
    if (!formData.subject.trim()) errs.subject = 'Subject line is required.';
    if (!formData.message.trim()) {
      errs.message = 'Inquiry message cannot be empty.';
    } else if (formData.message.trim().length < 15) {
      errs.message = 'Message must contain at least 15 characters.';
    }
    return errs;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const validationErrors = validate();
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }

    setErrors({});
    setSubmitted(true);
  };

  return (
    <div className="container" style={{ padding: '3.5rem 1.5rem 5rem', flex: 1, maxWidth: '680px' }}>
      <div style={{ marginBottom: '2rem', textAlign: 'center' }}>
        <span className="badge badge-accent" style={{ marginBottom: '0.75rem' }}>Direct Inquiries</span>
        <h1 style={{ fontSize: '2.2rem', marginBottom: '0.5rem' }}>Connect with Engineering</h1>
        <p style={{ color: 'var(--text-muted)' }}>
          Submit questions regarding deep learning deployment, camera calibration, or custom mill integration.
        </p>
      </div>

      {submitted ? (
        <div className="card" style={{ padding: '2.5rem', textAlign: 'center' }}>
          <div style={{
            width: '56px',
            height: '56px',
            borderRadius: '50%',
            background: 'rgba(69, 212, 154, 0.1)',
            border: '1px solid var(--success)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 1.25rem',
            color: 'var(--success)'
          }}>
            <CheckCircle2 size={30} />
          </div>
          <h2 style={{ fontSize: '1.4rem', marginBottom: '0.75rem' }}>Inquiry Registered</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem', marginBottom: '1.5rem', lineHeight: 1.6 }}>
            Thank you, <strong>{formData.name}</strong>. Your feedback has been verified in the front-end simulation interface. (Note: Email delivery is not configured on this standalone instance).
          </p>
          <button
            onClick={() => {
              setSubmitted(false);
              setFormData({ name: '', email: '', subject: '', message: '' });
            }}
            className="btn btn-secondary"
          >
            Submit Another Inquiry
          </button>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="card" style={{ padding: '2rem' }}>
          <div style={{ marginBottom: '1.25rem' }}>
            <label style={{ display: 'block', fontSize: '0.88rem', fontWeight: 600, marginBottom: '0.4rem' }}>
              Full Name *
            </label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              placeholder="e.g. Elena Rostova"
              style={{
                width: '100%',
                padding: '0.75rem',
                borderRadius: '6px',
                background: 'var(--surface-elevated)',
                border: `1px solid ${errors.name ? 'var(--danger)' : 'var(--border)'}`,
                color: 'var(--text)',
                fontFamily: 'inherit',
                fontSize: '0.95rem'
              }}
            />
            {errors.name && <div style={{ color: 'var(--danger)', fontSize: '0.78rem', marginTop: '0.3rem' }}>{errors.name}</div>}
          </div>

          <div style={{ marginBottom: '1.25rem' }}>
            <label style={{ display: 'block', fontSize: '0.88rem', fontWeight: 600, marginBottom: '0.4rem' }}>
              Corporate Email *
            </label>
            <input
              type="email"
              value={formData.email}
              onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              placeholder="e.g. e.rostova@steelforge.com"
              style={{
                width: '100%',
                padding: '0.75rem',
                borderRadius: '6px',
                background: 'var(--surface-elevated)',
                border: `1px solid ${errors.email ? 'var(--danger)' : 'var(--border)'}`,
                color: 'var(--text)',
                fontFamily: 'inherit',
                fontSize: '0.95rem'
              }}
            />
            {errors.email && <div style={{ color: 'var(--danger)', fontSize: '0.78rem', marginTop: '0.3rem' }}>{errors.email}</div>}
          </div>

          <div style={{ marginBottom: '1.25rem' }}>
            <label style={{ display: 'block', fontSize: '0.88rem', fontWeight: 600, marginBottom: '0.4rem' }}>
              Subject Line *
            </label>
            <input
              type="text"
              value={formData.subject}
              onChange={(e) => setFormData({ ...formData, subject: e.target.value })}
              placeholder="e.g. Model inference on 1.2mm gauge strip"
              style={{
                width: '100%',
                padding: '0.75rem',
                borderRadius: '6px',
                background: 'var(--surface-elevated)',
                border: `1px solid ${errors.subject ? 'var(--danger)' : 'var(--border)'}`,
                color: 'var(--text)',
                fontFamily: 'inherit',
                fontSize: '0.95rem'
              }}
            />
            {errors.subject && <div style={{ color: 'var(--danger)', fontSize: '0.78rem', marginTop: '0.3rem' }}>{errors.subject}</div>}
          </div>

          <div style={{ marginBottom: '1.75rem' }}>
            <label style={{ display: 'block', fontSize: '0.88rem', fontWeight: 600, marginBottom: '0.4rem' }}>
              Inquiry / Message *
            </label>
            <textarea
              rows={5}
              value={formData.message}
              onChange={(e) => setFormData({ ...formData, message: e.target.value })}
              placeholder="Specify rolling conditions, camera hardware, and deployment objectives..."
              style={{
                width: '100%',
                padding: '0.75rem',
                borderRadius: '6px',
                background: 'var(--surface-elevated)',
                border: `1px solid ${errors.message ? 'var(--danger)' : 'var(--border)'}`,
                color: 'var(--text)',
                fontFamily: 'inherit',
                fontSize: '0.95rem',
                resize: 'vertical'
              }}
            />
            {errors.message && <div style={{ color: 'var(--danger)', fontSize: '0.78rem', marginTop: '0.3rem' }}>{errors.message}</div>}
          </div>

          <button type="submit" className="btn btn-primary" style={{ width: '100%', padding: '0.85rem' }}>
            <Send size={18} /> Submit Technical Inquiry
          </button>
        </form>
      )}
    </div>
  );
}
