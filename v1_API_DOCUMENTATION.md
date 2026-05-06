# AMER-Net API Documentation

## Endpoints

### POST /analyze
Analyzes emotions from multimodal input.

**Request:**
\`\`\`bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@video.mp4" \
  -F "modality=video"
\`\`\`

**Response:**
\`\`\`json
{
  "request_id": "req_abc123",
  "predicted_emotion": "happiness",
  "confidence": 0.87,
  "modality_weights": {
    "text": 0.60,
    "vision": 0.30,
    "speech": 0.10
  },
  "processing_time_ms": 342.15,
  "timestamp": "2026-05-06T12:34:56"
}
\`\`\`

### GET /health
Returns API health status.

\`\`\`bash
curl http://localhost:8000/health
\`\`\`

## Error Handling
- 400: Bad request (missing parameters)
- 503: Service unavailable (models not initialized)
- 500: Internal server error