import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptivePreDecoder(nn.Module):
    """
    Main Version - Adaptive Pre-Decoder for Surface Code
    Novelty: Confidence-aware local error correction
    """
    def __init__(self, in_channels=1, hidden_dim=96, spatial_kernel=5):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 3D Convolutional Feature Extractor
        self.conv_net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Residual Correction Head
        self.correction_head = nn.Conv3d(hidden_dim, 1, kernel_size=1)
        
        # Confidence Prediction Head
        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
        self.spatial_kernel = spatial_kernel

    def forward(self, x):
        """
        x: [B, 1, X, Y, T]
        Returns:
            corrected: corrected syndrome
            confidence: confidence score [B, 1]
            residual: correction applied
        """
        residual = x
        
        features = self.conv_net(x)
        correction = self.correction_head(features)
        corrected = residual - correction
        
        confidence = self.confidence_head(features)
        
        return corrected, confidence.squeeze(-1), correction
