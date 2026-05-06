"""GA2MIF: Graph and Cross-Attention Multimodal Interaction Fusion Model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    VISUAL_HIDDEN_DIM, AUDIO_HIDDEN_DIM, TEXT_HIDDEN_DIM,
    FUSION_HIDDEN_DIM, NUM_EMOTIONS, ATTENTION_HEADS, DROPOUT_RATE
)

class MultimodalCrossAttention(nn.Module):
    """MPCAT: Multimodal Cross-Attention Layer."""
    
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(DROPOUT_RATE)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear transformations
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Concatenate heads
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        output = self.fc_out(context)
        
        return output, attention_weights

class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer for modality interactions."""
    
    def __init__(self, in_features, out_features, num_heads=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        
        self.attention = MultimodalCrossAttention(in_features, num_heads)
        self.norm = nn.LayerNorm(in_features)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, in_features * 4),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(in_features * 4, in_features)
        )
    
    def forward(self, x, adj_matrix=None):
        # Self-attention with normalization
        attended, weights = self.attention(x, x, x, mask=adj_matrix)
        x = self.norm(x + attended)
        
        # Feed-forward network
        x = self.norm(x + self.mlp(x))
        
        return x, weights

class GA2MIF(nn.Module):
    """Graph and Cross-Attention Multimodal Interaction Fusion Model."""
    
    def __init__(self, num_emotions=NUM_EMOTIONS):
        super().__init__()
        
        self.num_emotions = num_emotions
        self.fusion_dim = FUSION_HIDDEN_DIM
        
        # Feature projectors
        self.visual_proj = nn.Sequential(
            nn.Linear(VISUAL_HIDDEN_DIM, self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(AUDIO_HIDDEN_DIM, self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(TEXT_HIDDEN_DIM, self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )
        
        # Graph attention layers
        self.graph_layers = nn.ModuleList([
            GraphAttentionLayer(self.fusion_dim, self.fusion_dim, ATTENTION_HEADS)
            for _ in range(2)
        ])
        
        # Cross-attention layers
        self.cross_attention_visual = MultimodalCrossAttention(self.fusion_dim, ATTENTION_HEADS)
        self.cross_attention_audio = MultimodalCrossAttention(self.fusion_dim, ATTENTION_HEADS)
        self.cross_attention_text = MultimodalCrossAttention(self.fusion_dim, ATTENTION_HEADS)
        
        # Fusion layer
        self.fusion_fc = nn.Sequential(
            nn.Linear(self.fusion_dim * 3, self.fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),
            nn.ReLU()
        )
        
        # Classification head
        self.classifier = nn.Linear(self.fusion_dim, num_emotions)
    
    def forward(self, visual_features=None, audio_features=None, text_features=None, modality_weights=None):
        """
        Forward pass through GA2MIF.
        
        Args:
            visual_features: (B, T, VISUAL_HIDDEN_DIM) or None
            audio_features: (B, T, AUDIO_HIDDEN_DIM) or None
            text_features: (B, TEXT_HIDDEN_DIM) or None
            modality_weights: Dict with keys 'text', 'vision', 'speech'
        
        Returns:
            emotion_logits: (B, NUM_EMOTIONS)
            attention_maps: Dict with attention weights
        """
        
        # Project features to fusion dimension
        if visual_features is not None:
            visual_proj = self.visual_proj(visual_features)
        else:
            visual_proj = None
        
        if audio_features is not None:
            audio_proj = self.audio_proj(audio_features)
        else:
            audio_proj = None
        
        if text_features is not None:
            text_proj = self.text_proj(text_features.unsqueeze(1))  # (B, 1, FUSION_DIM)
        else:
            text_proj = None
        
        # Apply gated weighting if provided
        if modality_weights:
            if visual_proj is not None:
                visual_proj = visual_proj * modality_weights.get('vision', 0.33)
            if audio_proj is not None:
                audio_proj = audio_proj * modality_weights.get('speech', 0.33)
            if text_proj is not None:
                text_proj = text_proj * modality_weights.get('text', 0.33)
        
        # Graph attention processing
        attention_maps = {}
        
        # Build node representation (concatenate available modalities)
        nodes = []
        if visual_proj is not None:
            nodes.append(visual_proj.mean(dim=1))  # (B, FUSION_DIM)
        if audio_proj is not None:
            nodes.append(audio_proj.mean(dim=1))   # (B, FUSION_DIM)
        if text_proj is not None:
            nodes.append(text_proj.squeeze(1))     # (B, FUSION_DIM)
        
        if not nodes:
            raise ValueError("At least one modality must be provided")
        
        # Stack nodes
        graph_nodes = torch.stack(nodes, dim=1)  # (B, num_modalities, FUSION_DIM)
        
        # Apply graph attention layers
        for i, layer in enumerate(self.graph_layers):
            graph_nodes, weights = layer(graph_nodes)
            attention_maps[f'graph_layer_{i}'] = weights
        
        # Cross-attention between modalities
        if visual_proj is not None and text_proj is not None:
            visual_attended, visual_attention = self.cross_attention_visual(
                visual_proj.mean(dim=1).unsqueeze(1),
                text_proj,
                text_proj
            )
            attention_maps['visual_text_cross'] = visual_attention
            visual_feat = visual_attended.squeeze(1)
        elif visual_proj is not None:
            visual_feat = visual_proj.mean(dim=1)
        else:
            visual_feat = torch.zeros(graph_nodes.size(0), self.fusion_dim, device=graph_nodes.device)
        
        if audio_proj is not None and text_proj is not None:
            audio_attended, audio_attention = self.cross_attention_audio(
                audio_proj.mean(dim=1).unsqueeze(1),
                text_proj,
                text_proj
            )
            attention_maps['audio_text_cross'] = audio_attention
            audio_feat = audio_attended.squeeze(1)
        elif audio_proj is not None:
            audio_feat = audio_proj.mean(dim=1)
        else:
            audio_feat = torch.zeros(graph_nodes.size(0), self.fusion_dim, device=graph_nodes.device)
        
        text_feat = text_proj.squeeze(1) if text_proj is not None else torch.zeros(graph_nodes.size(0), self.fusion_dim, device=graph_nodes.device)
        
        # Fuse all modalities
        fused = torch.cat([visual_feat, audio_feat, text_feat], dim=1)
        fused = self.fusion_fc(fused)
        
        # Classify
        emotion_logits = self.classifier(fused)
        
        return emotion_logits, attention_maps