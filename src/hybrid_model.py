import torch
import torch.nn as nn
from .pre_decoder import AdaptivePreDecoder
from .transformer_models import SpatioTemporalLocalTransformer

class AdaptiveHybridDecoder(nn.Module):
    """
    Main Hybrid Model (Final Version)
    Combines Adaptive Pre-Decoder + Spatio-Temporal Transformer
    """
    def __init__(self, d_model=240, num_layers=4, confidence_threshold=0.65):
        super().__init__()
        
        self.pre_decoder = AdaptivePreDecoder()
        self.st_transformer = SpatioTemporalLocalTransformer(
            d_model=d_model,
            num_layers=num_layers
        )
        
        self.confidence_threshold = confidence_threshold
        
    def forward(self, x):
        """
        x: [B, 1, X, Y, T]
        """
        # Stage 1: Adaptive Pre-Decoding
        corrected, confidence, _ = self.pre_decoder(x)
        
        # Stage 2: Confidence-based Routing
        batch_size = x.shape[0]
        high_conf_mask = (confidence > self.confidence_threshold).view(batch_size, 1, 1, 1, 1)
        
        # Use corrected version only where confidence is high
        routed_input = torch.where(high_conf_mask, corrected, x)
        
        # Stage 3: Main Spatio-Temporal Transformer
        logits = self.st_transformer(routed_input)
        
        return {
            'logits': logits,
            'confidence': confidence,
            'high_conf_ratio': high_conf_mask.float().mean()
        }

    def get_model_info(self):
        return {
            "model": "AdaptiveHybridDecoder",
            "pre_decoder": "3D CNN Adaptive",
            "main_decoder": "Spatio-Temporal Local Transformer",
            "confidence_threshold": self.confidence_threshold,
            "total_parameters": sum(p.numel() for p in self.parameters())
        }


# Quick Test
if __name__ == "__main__":
    model = AdaptiveHybridDecoder()
    dummy_input = torch.randn(4, 1, 12, 12, 7)
    output = model(dummy_input)
    print("Hybrid Model Output Keys:", output.keys())
    print("Logits shape:", output['logits'].shape)
    print("Avg Confidence:", output['confidence'].mean().item())






