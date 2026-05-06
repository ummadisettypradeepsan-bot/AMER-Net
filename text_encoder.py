"""Text feature encoder using Bidirectional LSTM with attention."""
import torch
import torch.nn as nn
from config import TEXT_HIDDEN_DIM

class AttentionLayer(nn.Module):
    """Self-attention layer for text encoding."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        
        context = torch.matmul(weights, V)
        return context, weights

class TextEncoder(nn.Module):
    """Bidirectional LSTM with self-attention for text encoding."""
    
    def __init__(self, vocab_size=5000, embedding_dim=100, hidden_dim=TEXT_HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Attention
        self.attention = AttentionLayer(hidden_dim * 2)
        
        # Output projection
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x: (B, seq_len) - token indices
            lengths: (B,) - sequence lengths
        
        Returns:
            features: (B, hidden_dim)
        """
        # Embedding
        embedded = self.embedding(x)  # (B, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded)  # (B, seq_len, hidden_dim*2)
        
        # Attention
        context, attention_weights = self.attention(lstm_out)
        
        # Global average pooling
        if lengths is not None:
            mask = self._create_mask(lengths, lstm_out.device)
            context = context * mask.unsqueeze(-1)
            context = context.sum(dim=1) / lengths.unsqueeze(-1).float()
        else:
            context = context.mean(dim=1)
        
        # Project
        features = self.projection(context)  # (B, hidden_dim)
        
        return features
    
    def _create_mask(self, lengths, device):
        """Create attention mask from sequence lengths."""
        batch_size, max_len = len(lengths), lengths.max().item()
        mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        return mask.float()