"""Configuration management for AMER-Net backend."""
import os
from pathlib import Path
import torch
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models" / "weights"
LOG_DIR = BASE_DIR / "logs"

# Create directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Model Configuration
VISUAL_MODEL_PATH = str(MODEL_DIR / "densenet121.pt")
AUDIO_MODEL_PATH = str(MODEL_DIR / "lstm_audio.pt")
TEXT_MODEL_PATH = str(MODEL_DIR / "lstm_text.pt")
FUSION_MODEL_PATH = str(MODEL_DIR / "ga2mif.pt")

# Processing Configuration
VIDEO_FPS = 30
AUDIO_SAMPLE_RATE = 16000
TEXT_MAX_LENGTH = 512
CHUNK_DURATION = 1.0  # seconds

# Model Architecture
VISUAL_HIDDEN_DIM = 256
AUDIO_HIDDEN_DIM = 128
TEXT_HIDDEN_DIM = 256
FUSION_HIDDEN_DIM = 512
NUM_EMOTIONS = 7  # angry, disgust, fear, happy, neutral, sad, surprise

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Training/Inference Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
L2_LAMBDA = 1e-4
BATCH_SIZE = 32

# API Configuration
API_TITLE = "AMER-Net: Adaptive Multimodal Emotion Recognition"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
Advanced AI system for emotion recognition using multimodal fusion of visual, 
acoustic, and textual data with gated attention mechanisms.
"""

# CORS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")

# Audio Processing
AUDIO_SNR_THRESHOLD = 15.0  # dB (below this, reduce audio weight)
AUDIO_MIN_WEIGHT = 0.1
AUDIO_MAX_WEIGHT = 0.4

# Text Processing
TEXT_MIN_WEIGHT = 0.4
TEXT_MAX_WEIGHT = 0.7

# Vision Processing
VISION_MIN_WEIGHT = 0.2
VISION_MAX_WEIGHT = 0.5

# Attention Configuration
ATTENTION_HEADS = 8
DROPOUT_RATE = 0.3

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")