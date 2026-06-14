import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptivePreDecoder(nn.Module):
    """
    Lightweight Adaptive Pre-Decoder for Surface Code
    - Uses 3D CNN for local error correction
    - Outputs confidence score + residual syndrome
    """
    
    def __init__(self, in_channels=1, hidden_dim=64, spatial_radius=3):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1)
        
        self.residual_conv = nn.Conv3d(hidden_dim, 1, kernel_size=1)
        
        # Confidence prediction head
        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.spatial_radius = spatial_radius

    def forward(self, x):
        """
        x: [B, 1, X, Y, T]  -> 3D spatio-temporal tensor
        """
        residual = x.clone()
        
        # Feature extraction
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        
        # Local correction
        correction = self.residual_conv(h)
        corrected = residual - correction  # simple residual learning
        
        # Confidence score (how much we trust the correction)
        confidence = self.confidence_head(h)
        
        return corrected, confidence, correction


# For testing
if __name__ == "__main__":
    model = AdaptivePreDecoder()
    dummy_input = torch.randn(4, 1, 10, 10, 5)  # B, C, X, Y, T
    corrected, conf, _ = model(dummy_input)
    print("Pre-decoder output shapes:")
    print("Corrected:", corrected.shape)
    print("Confidence:", conf.shape)
