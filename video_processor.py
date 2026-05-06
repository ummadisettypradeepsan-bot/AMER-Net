"""Video processing utilities for frame extraction and preprocessing."""
import cv2
import numpy as np
from pathlib import Path
import tempfile
import logging
from config import VIDEO_FPS, VISUAL_HIDDEN_DIM

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Process video files to extract frames."""
    
    def __init__(self, target_fps=VIDEO_FPS, frame_size=(224, 224)):
        self.target_fps = target_fps
        self.frame_size = frame_size
    
    async def process_video(self, upload_file):
        """
        Process uploaded video file and extract frames.
        
        Args:
            upload_file: FastAPI UploadFile object
        
        Returns:
            visual_features: (B, T, VISUAL_HIDDEN_DIM) - dummy features
            audio_features: (B, T, AUDIO_HIDDEN_DIM) - dummy features
            timestamps: List of frame timestamps
        """
        try:
            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                contents = await upload_file.read()
                tmp.write(contents)
                tmp_path = tmp.name
            
            # Open video
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {upload_file.filename}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames at target FPS
            frames = []
            timestamps = []
            frame_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize and normalize
                frame = cv2.resize(frame, self.frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
                
                timestamp = frame_idx / fps if fps > 0 else 0
                timestamps.append(timestamp)
                
                frame_idx += 1
                
                # Limit to 300 frames (~10 seconds at 30 FPS)
                if len(frames) >= 300:
                    break
            
            cap.release()
            Path(tmp_path).unlink()  # Delete temp file
            
            if not frames:
                raise ValueError("No frames extracted from video")
            
            logger.info(f"Extracted {len(frames)} frames from video")
            
            # Convert to numpy array
            frames_array = np.array(frames)  # (T, 224, 224, 3)
            frames_array = np.transpose(frames_array, (3, 0, 1, 2))  # (3, T, 224, 224)
            frames_array = np.expand_dims(frames_array, 0)  # (B=1, 3, T, 224, 224)
            
            # Dummy feature extraction (replace with actual model)
            visual_features = np.random.randn(1, len(frames), VISUAL_HIDDEN_DIM).astype(np.float32)
            audio_features = np.random.randn(1, len(frames), 128).astype(np.float32)
            
            return visual_features, audio_features, timestamps
        
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise