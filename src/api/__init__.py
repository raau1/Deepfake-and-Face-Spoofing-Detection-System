"""
FastAPI web deployment for the deepfake detection system.
Implements the web-deployment objective from Interim Report Section 1.3 and 4.2.1.

Exposes three endpoints:
    GET  /                 , HTML upload page (Jinja2 template)
    POST /api/predict      , accepts a video upload, returns a JSON verdict
    GET  /api/health       , health check, reports device and loaded models
    GET  /api/models       , lists available models with their evaluation metrics
"""
