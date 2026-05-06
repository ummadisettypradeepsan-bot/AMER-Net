"""Audio feature encoder using Bidirectional LSTM."""
import torch
import torch.nn as nn
from config import AUDIO_HIDDEN_DIM

class AudioEncoder(nn.Module):
    """Bidirectional LSTM based audio feature encoder."""
    
    def __init__(self, input_dim=128, hidden_dim=AUDIO_HIDDEN_DIM, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Output projection
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, T, input_dim) - acoustic features
        
        Returns:
            features: (B, T, hidden_dim)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)  # (B, T, hidden_dim*2)
        
        # Project to hidden dimension
        features = self.projection(lstm_out)  # (B, T, hidden_dim)
        
        return features