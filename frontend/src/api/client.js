export async function fetchHealth() {
  const res = await fetch('/api/health');
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

export async function fetchModelStatus() {
  const res = await fetch('/api/model-status');
  if (!res.ok) throw new Error('Model status fetch failed');
  return res.json();
}

export async function runInference(file, threshold = 0.5) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('threshold', threshold.toString());

  const res = await fetch('/api/predict', {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Inference error: HTTP ${res.status}`);
  }

  return res.json();
}
