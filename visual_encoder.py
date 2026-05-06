"""Visual feature encoder using DenseNet-121."""
import torch
import torch.nn as nn
import torchvision.models as models
from config import VISUAL_HIDDEN_DIM

class VisualEncoder(nn.Module):
    """DenseNet-121 based visual feature encoder."""
    
    def __init__(self, hidden_dim=VISUAL_HIDDEN_DIM, pretrained=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Load pretrained DenseNet-121
        densenet = models.densenet121(pretrained=pretrained)
        
        # Remove classifier
        self.features = densenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Get feature dimension
        feature_dim = densenet.classifier.in_features
        
        # Projection to hidden dimension
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Freeze backbone initially (can be unfrozen for fine-tuning)
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, T, 3, 224, 224) - batch of frames
        
        Returns:
            features: (B, T, hidden_dim)
        """
        batch_size, num_frames, _, _, _ = x.shape
        
        # Reshape to process all frames at once
        x = x.view(batch_size * num_frames, 3, 224, 224)
        
        # Extract features
        features = self.features(x)
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)
        
        # Project to hidden dimension
        features = self.projection(features)
        
        # Reshape back to temporal dimension
        features = features.view(batch_size, num_frames, self.hidden_dim)
        
        return features