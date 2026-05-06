"""Audio processing utilities for feature extraction."""
import librosa
import numpy as np
import tempfile
from pathlib import Path
import logging
from config import AUDIO_SAMPLE_RATE, AUDIO_SNR_THRESHOLD

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Process audio files to extract acoustic features."""
    
    def __init__(self, sr=AUDIO_SAMPLE_RATE, n_mfcc=13):
        self.sr = sr
        self.n_mfcc = n_mfcc
    
    async def process_audio(self, upload_file):
        """
        Process uploaded audio file and extract features.
        
        Args:
            upload_file: FastAPI UploadFile object
        
        Returns:
            audio_features: (B, T, n_mfcc) - MFCC features
        """
        try:
            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                contents = await upload_file.read()
                tmp.write(contents)
                tmp_path = tmp.name
            
            # Load audio
            y, sr = librosa.load(tmp_path, sr=self.sr)
            Path(tmp_path).unlink()
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            
            # Transpose to (T, n_mfcc)
            mfcc = mfcc.T
            
            # Add batch dimension
            audio_features = np.expand_dims(mfcc, 0)  # (1, T, n_mfcc)
            
            # Compute SNR for gated fusion
            snr_db = self._compute_snr(y)
            logger.info(f"Audio SNR: {snr_db:.2f} dB")
            
            return audio_features
        
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            raise
    
    def _compute_snr(self, y, frame_length=2048):
        """Compute signal-to-noise ratio."""
        # Estimate noise as the quietest frame
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=frame_length//2)
        frame_power = np.mean(frames ** 2, axis=0)
        
        noise_power = np.min(frame_power)
        signal_power = np.mean(frame_power)
        
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        return snr_db