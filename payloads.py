"""Pydantic schemas for request/response validation."""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime

class AnalysisRequest(BaseModel):
    """Request model for emotion analysis."""
    modality: str = Field(..., description="Type: 'video', 'audio', or 'text'")
    normalize_audio: bool = Field(default=True, description="Apply audio normalization")
    
    class Config:
        json_schema_extra = {
            "example": {
                "modality": "video",
                "normalize_audio": True
            }
        }

class ModalityWeights(BaseModel):
    """Modality attention weights."""
    text: float = Field(..., ge=0.0, le=1.0, description="Text modality weight")
    vision: float = Field(..., ge=0.0, le=1.0, description="Visual modality weight")
    speech: float = Field(..., ge=0.0, le=1.0, description="Audio modality weight")

class FrameEmotion(BaseModel):
    """Per-frame emotion prediction."""
    timestamp: float = Field(..., description="Timestamp in seconds")
    emotion: str = Field(..., description="Predicted emotion label")
    confidence: float = Field(..., ge=0.0, le=1.0)

class AnalysisResponse(BaseModel):
    """Response model for emotion analysis."""
    request_id: str = Field(..., description="Unique request identifier")
    predicted_emotion: str = Field(..., description="Primary emotion prediction")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    modality_weights: ModalityWeights = Field(..., description="Attention weights per modality")
    frame_emotions: List[FrameEmotion] = Field(default_factory=list, description="Per-frame predictions")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    timestamp: str = Field(..., description="ISO8601 timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_12345",
                "predicted_emotion": "happiness",
                "confidence": 0.87,
                "modality_weights": {
                    "text": 0.60,
                    "vision": 0.30,
                    "speech": 0.10
                },
                "frame_emotions": [
                    {"timestamp": 0.0, "emotion": "neutral", "confidence": 0.75},
                    {"timestamp": 0.5, "emotion": "happiness", "confidence": 0.89}
                ],
                "processing_time_ms": 342.15,
                "timestamp": "2026-05-06T12:34:56"
            }
        }

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status: 'healthy' or 'unhealthy'")
    models_loaded: Dict[str, bool] = Field(..., description="Status of each model")
    device: str = Field(..., description="Computation device: 'cuda' or 'cpu'")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="ISO8601 timestamp")