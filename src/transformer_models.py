import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional_encodings import SupraLaplacianEncoding  # পরে তৈরি করবো

class SpatioTemporalLocalTransformer(nn.Module):
    """
    Original Paper-এর মেইন মডেল (Factorized Spatio-Temporal Transformer)
    """
    def __init__(self, d_model=240, num_layers=4, spatial_heads=8, temporal_heads=4, 
                 ffn_dim=1024, dropout=0.1, spatial_radius=4, temporal_window=3):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.spatial_radius = spatial_radius
        self.temporal_window = temporal_window

        # Positional Encoding
        self.pos_encoding = SupraLaplacianEncoding(d_model)

        # Layers
        self.layers = nn.ModuleList([
            SpatioTemporalBlock(d_model, spatial_heads, temporal_heads, ffn_dim, dropout, 
                              spatial_radius, temporal_window)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, 1)   # Logical flip probability

    def forward(self, x):
        """
        x: [B, 1, X, Y, T]  -> 3D Spatio-Temporal input
        """
        B, _, X, Y, T = x.shape
        
        # Flatten + Embed
        x = x.view(B, X*Y*T, 1)  # [B, seq_len, 1]
        x = x * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=x.device))
        
        # Add positional encoding
        x = self.pos_encoding(x, X, Y, T)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        x = x.mean(dim=1)                    # Global pooling
        logits = self.output_head(x).squeeze(-1)
        
        return logits


class SpatioTemporalBlock(nn.Module):
    """Factorized Space + Time Attention Block"""
    def __init__(self, d_model, spatial_heads, temporal_heads, ffn_dim, dropout, 
                 spatial_radius=4, temporal_window=3):
        super().__init__()
        
        self.spatial_attn = nn.MultiheadAttention(d_model, spatial_heads, dropout=dropout, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(d_model, temporal_heads, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        residual = x
        
        # Spatial Attention
        x = self.norm1(x)
        attn_out, _ = self.spatial_attn(x, x, x)
        x = residual + attn_out
        
        residual = x
        # Temporal Attention
        x = self.norm2(x)
        attn_out, _ = self.temporal_attn(x, x, x)
        x = residual + attn_out
        
        # FFN
        residual = x
        x = self.norm3(x)
        x = residual + self.ffn(x)
        
        return x
