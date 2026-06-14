import torch
import yaml
from pathlib import Path

def load_config(config_path="configs/config.yaml"):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """Track average values during training"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename="checkpoint.pth"):
    """Save model checkpoint"""
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")


# For later integration with original circuit.py
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Utils loaded successfully")
