"""Main FastAPI application for AMER-Net."""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import uuid
from datetime import datetime
import torch

from config import (
    API_TITLE, API_VERSION, API_DESCRIPTION, ALLOWED_ORIGINS, 
    DEVICE, MODEL_DIR, LOG_DIR
)
from schemas.payloads import AnalysisRequest, AnalysisResponse, HealthResponse
from models.ga2mif import GA2MIF
from utils.video_processor import VideoProcessor
from utils.audio_processor import AudioProcessor
from utils.fusion_logic import GatedFusionLogic

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model container
class ModelManager:
    def __init__(self):
        self.fusion_model = None
        self.visual_processor = None
        self.audio_processor = None
        self.text_processor = None
        self.fusion_logic = None
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize all models and processors."""
        try:
            logger.info("Initializing GA2MIF fusion model...")
            self.fusion_model = GA2MIF()
            self.fusion_model.to(DEVICE)
            self.fusion_model.eval()
            
            logger.info("Initializing processors...")
            self.visual_processor = VideoProcessor()
            self.audio_processor = AudioProcessor()
            self.text_processor = TextProcessor()
            self.fusion_logic = GatedFusionLogic()
            
            self.is_initialized = True
            logger.info("✓ All models initialized successfully")
        except Exception as e:
            logger.error(f"✗ Initialization failed: {str(e)}")
            self.is_initialized = False
            raise

model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    await model_manager.initialize()
    logger.info("AMER-Net backend started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("AMER-Net backend shutting down")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint. Returns model status and availability.
    
    Returns:
        HealthResponse: Status of all models and components
    """
    return HealthResponse(
        status="healthy" if model_manager.is_initialized else "unhealthy",
        models_loaded={
            "visual": model_manager.visual_processor is not None,
            "audio": model_manager.audio_processor is not None,
            "text": model_manager.text_processor is not None,
            "fusion": model_manager.fusion_model is not None,
        },
        device=DEVICE,
        version=API_VERSION,
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_emotion(
    file: UploadFile = File(None),
    text: str = None,
    modality: str = "video"
):
    """
    Analyze emotions from multimodal input (video, audio, or text).
    
    Args:
        file: Uploaded video or audio file (multipart)
        text: Text input as JSON string
        modality: Type of input ('video', 'audio', or 'text')
    
    Returns:
        AnalysisResponse: Emotion prediction with attention weights
    
    Raises:
        HTTPException: If analysis fails
    """
    if not model_manager.is_initialized:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"[{request_id}] Analyzing {modality} input")
        
        # Extract features based on modality
        if modality == "video" and file:
            visual_features, audio_features, timestamps = await model_manager.visual_processor.process_video(file)
            text_features = None
        elif modality == "audio" and file:
            audio_features = await model_manager.audio_processor.process_audio(file)
            visual_features = None
            text_features = None
        elif modality == "text" and text:
            text_features = model_manager.text_processor.process_text(text)
            visual_features = None
            audio_features = None
        else:
            raise HTTPException(status_code=400, detail="Invalid modality or missing input")
        
        # Apply gated fusion logic
        modality_weights = model_manager.fusion_logic.compute_weights(
            visual_features=visual_features,
            audio_features=audio_features,
            text_features=text_features
        )
        
        logger.info(f"[{request_id}] Modality weights: {modality_weights}")
        
        # Run inference through GA2MIF
        with torch.no_grad():
            if visual_features is not None:
                visual_features = torch.from_numpy(visual_features).to(DEVICE).float()
            if audio_features is not None:
                audio_features = torch.from_numpy(audio_features).to(DEVICE).float()
            if text_features is not None:
                text_features = torch.from_numpy(text_features).to(DEVICE).float()
            
            emotion_logits, attention_maps = model_manager.fusion_model(
                visual_features=visual_features,
                audio_features=audio_features,
                text_features=text_features,
                modality_weights=modality_weights
            )
        
        # Process results
        emotion_probs = torch.softmax(emotion_logits, dim=1)
        confidence, predicted_emotion_idx = torch.max(emotion_probs, dim=1)
        predicted_emotion = EMOTION_LABELS[predicted_emotion_idx.item()]
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        logger.info(f"[{request_id}] ✓ Analysis complete: {predicted_emotion} ({confidence:.2%}) in {processing_time:.2f}ms")
        
        return AnalysisResponse(
            request_id=request_id,
            predicted_emotion=predicted_emotion,
            confidence=float(confidence.item()),
            modality_weights=modality_weights,
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        logger.error(f"[{request_id}] ✗ Analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)