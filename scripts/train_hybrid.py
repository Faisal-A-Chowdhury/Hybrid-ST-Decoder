import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hybrid_model import AdaptiveHybridDecoder
from src.utils import load_config, AverageMeter, get_device, count_parameters, save_checkpoint
from src.circuit import get_data_surface, transform_to_3d

def main():
    config = load_config("configs/config.yaml")
    device = get_device()
    print(f"Using device: {device}\n")

    # ==================== Model ====================
    model = AdaptiveHybridDecoder(
        d_model=config['model']['d_model'],
        num_layers=config['model']['num_layers'],
        confidence_threshold=config['model']['confidence_threshold']
    ).to(device)

    print(f"Model Loaded - Total Parameters: {count_parameters(model):,}\n")

    # ==================== Optimizer & Loss ====================
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config['training']['learning_rate'],
                           weight_decay=config['training']['weight_decay'])

    # ==================== Real Data Loading ====================
    print("Generating Real Surface Code Data using Stim...")
    d = config['data']['distance']
    r = config['data']['rounds']
    p = config['data']['error_rate']
    
    detections, flips, num_input = get_data_surface(
        d=d, 
        r=r, 
        p=p, 
        train_size=config['data']['train_size']
    )

    # Convert to 3D Spatio-Temporal Tensor
    X_train = transform_to_3d(detections, d, r)
    y_train = flips

    print(f"Data Loaded -> Input Shape: {X_train.shape} | Labels Shape: {y_train.shape}")

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # ==================== Training Loop ====================
    print("\n=== Starting Training ===\n")
    model.train()

    for epoch in range(config['training']['epochs']):
        loss_meter = AverageMeter()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output['logits'], target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            optimizer.step()
            
            loss_meter.update(loss.item())
            
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{config['training']['epochs']}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss_meter.avg:.6f} | "
                      f"Conf: {output['confidence'].mean().item():.4f}")

        print(f"✅ Epoch {epoch+1} Completed - Avg Loss: {loss_meter.avg:.6f}\n")

    # Save Final Model
    print("Training Finished! Saving model...")
    save_checkpoint({
        'epoch': config['training']['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, filename="results/checkpoints/hybrid_final.pth")

    print("🎉 Training Completed Successfully!")


if __name__ == "__main__":
    main()
