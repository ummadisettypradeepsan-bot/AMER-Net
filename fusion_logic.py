"""Gated fusion logic for adaptive modality weighting."""
import numpy as np
import logging
from config import (
    AUDIO_SNR_THRESHOLD, AUDIO_MIN_WEIGHT, AUDIO_MAX_WEIGHT,
    TEXT_MIN_WEIGHT, TEXT_MAX_WEIGHT,
    VISION_MIN_WEIGHT, VISION_MAX_WEIGHT
)

logger = logging.getLogger(__name__)

class GatedFusionLogic:
    """Implements gated fusion logic for modality weighting."""
    
    def compute_weights(self, visual_features=None, audio_features=None, text_features=None):
        """
        Compute adaptive modality weights based on signal quality.
        
        Args:
            visual_features: Visual feature array
            audio_features: Audio feature array
            text_features: Text feature array
        
        Returns:
            Dict with keys 'text', 'vision', 'speech' summing to 1.0
        """
        weights = {'text': 0.0, 'vision': 0.0, 'speech': 0.0}
        num_modalities = 0
        
        # Check which modalities are available
        has_visual = visual_features is not None
        has_audio = audio_features is not None
        has_text = text_features is not None
        
        if not (has_visual or has_audio or has_text):
            raise ValueError("At least one modality must be provided")
        
        # Compute base weights based on modality availability
        if has_visual:
            weights['vision'] = VISION_MIN_WEIGHT + (VISION_MAX_WEIGHT - VISION_MIN_WEIGHT) * 0.5
            num_modalities += 1
        
        if has_audio:
            # Gated speech fusion: reduce weight if SNR is poor
            snr_db = self._estimate_snr(audio_features)
            if snr_db < AUDIO_SNR_THRESHOLD:
                logger.warning(f"Low audio SNR: {snr_db:.2f} dB (threshold: {AUDIO_SNR_THRESHOLD})")
                weights['speech'] = AUDIO_MIN_WEIGHT
            else:
                weights['speech'] = AUDIO_MIN_WEIGHT + (AUDIO_MAX_WEIGHT - AUDIO_MIN_WEIGHT) * 0.7
            num_modalities += 1
        
        if has_text:
            # Text priority: boost text weight as it provides strong auxiliary signal
            weights['text'] = TEXT_MIN_WEIGHT + (TEXT_MAX_WEIGHT - TEXT_MIN_WEIGHT) * 0.8
            num_modalities += 1
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        for key in weights:
            if total_weight > 0:
                weights[key] = weights[key] / total_weight
        
        logger.info(f"Computed modality weights: {weights}")
        return weights
    
    def _estimate_snr(self, audio_features):
        """Estimate SNR from audio features."""
        if audio_features is None or len(audio_features) == 0:
            return 0.0
        
        # Simple SNR estimation from feature variance
        signal_var = np.var(audio_features)
        noise_var = np.var(np.diff(audio_features, axis=1))
        
        snr_db = 10 * np.log10(signal_var / (noise_var + 1e-10))
        return snr_db