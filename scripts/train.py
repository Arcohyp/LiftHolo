import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import yaml

from liftHolo import LiftHolo


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class HologramDataset(Dataset):
    """
    Dataset for hologram training.
    
    Expected directory structure:
        data_dir/
            ├── hr/          # High-resolution amplitude images
            │   ├── 0001.png
            │   └── ...
            └── lr/          # Low-resolution amplitude images (downsampled)
                ├── 0001.png
                └── ...
    
    Note: You need to prepare your own dataset or use public datasets like DIV2K.
    For hologram generation, the "HR" here refers to the target amplitude,
    and "LR" is the downsampled input amplitude.
    """
    def __init__(self, data_dir, hr_size=(2160, 3840), lr_size=(1080, 1920), scale_factor=2):
        self.data_dir = data_dir
        self.hr_dir = os.path.join(data_dir, 'hr')
        self.lr_dir = os.path.join(data_dir, 'lr')
        
        # Get list of images
        if os.path.exists(self.hr_dir):
            self.image_files = sorted([f for f in os.listdir(self.hr_dir) 
                                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        else:
            self.image_files = []
            print(f"Warning: {self.hr_dir} not found. Please prepare your dataset.")
        
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.scale_factor = scale_factor
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load HR amplitude
        hr_path = os.path.join(self.hr_dir, img_name)
        hr_img = Image.open(hr_path).convert('L')
        hr_img = hr_img.resize((self.hr_size[1], self.hr_size[0]), Image.BICUBIC)
        hr_amp = np.array(hr_img).astype(np.float32) / 255.0
        
        # Load LR amplitude (or downsample HR)
        lr_path = os.path.join(self.lr_dir, img_name)
        if os.path.exists(lr_path):
            lr_img = Image.open(lr_path).convert('L')
            lr_img = lr_img.resize((self.lr_size[1], self.lr_size[0]), Image.BICUBIC)
            lr_amp = np.array(lr_img).astype(np.float32) / 255.0
        else:
            # Downsample HR if LR doesn't exist
            lr_amp = hr_amp[::self.scale_factor, ::self.scale_factor]
        
        # Convert to tensors
        hr_tensor = torch.from_numpy(hr_amp).unsqueeze(0)  # [1, H, W]
        lr_tensor = torch.from_numpy(lr_amp).unsqueeze(0)  # [1, h, w]
        
        return {
            'lr_amp': lr_tensor,
            'hr_amp': hr_tensor,
            'filename': img_name
        }


def train_one_epoch(model, dataloader, optimizer, device, config):
    model.train()
    total_loss = 0.0
    
    # Optical parameters from paper
    pitch = config['optical']['pitch']
    wavelength = config['optical']['wavelength']
    z = config['optical']['z']
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        lr_amp = batch['lr_amp'].to(device)
        hr_amp = batch['hr_amp'].to(device)
        
        # Create zero phase for initialization
        phase = torch.zeros_like(lr_amp)
        
        # Forward pass
        optimizer.zero_grad()
        
        holophase, sr_amp = model(
            lr_amp, phase, z, 
            model.prop_flag, 
            pitch, wavelength, 
            hr_amp
        )
        
        # Loss: MSE between reconstructed amplitude and target
        # Note: For full hologram training, you need to compute the 
        # reconstructed image via ASM propagation and compare with target
        loss = F.mse_loss(sr_amp, hr_amp)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device, config):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    
    pitch = config['optical']['pitch']
    wavelength = config['optical']['wavelength']
    z = config['optical']['z']
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            lr_amp = batch['lr_amp'].to(device)
            hr_amp = batch['hr_amp'].to(device)
            phase = torch.zeros_like(lr_amp)
            
            holophase, sr_amp = model(
                lr_amp, phase, z,
                model.prop_flag,
                pitch, wavelength,
                hr_amp
            )
            
            loss = F.mse_loss(sr_amp, hr_amp)
            total_loss += loss.item()
            
            # Simple PSNR calculation
            mse = F.mse_loss(sr_amp, hr_amp)
            psnr = 10 * torch.log10(1.0 / mse) if mse > 0 else torch.tensor(100.0)
            total_psnr += psnr.item()
    
    return total_loss / len(dataloader), total_psnr / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train LiftHolo model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--val-dir', type=str, default=None,
                        help='Path to validation data directory')
    parser.add_argument('--output-dir', type=str, default='./experiments',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = LiftHolo(
        scale_factor=config['model']['scale_factor'],
        amp_channels=config['model']['amp_channels']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    train_dataset = HologramDataset(
        args.data_dir,
        scale_factor=config['model']['scale_factor']
    )
    
    if len(train_dataset) == 0:
        print("\n" + "="*60)
        print("ERROR: No training data found!")
        print("="*60)
        print("\nPlease prepare your dataset with the following structure:")
        print(f"  {args.data_dir}/")
        print(f"    ├── hr/     # High-resolution amplitude images")
        print(f"    └── lr/     # Low-resolution amplitude images")
        print("\nYou can use public datasets like DIV2K and downsample them.")
        print("="*60 + "\n")
        return
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = None
    if args.val_dir:
        val_dataset = HologramDataset(
            args.val_dir,
            scale_factor=config['model']['scale_factor']
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2
        )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float('inf')
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}\n")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, device, config)
        print(f"Train Loss: {train_loss:.6f}")
        
        if val_loader:
            val_loss, val_psnr = validate(model, val_loader, device, config)
            print(f"Val Loss: {val_loss:.6f}, Val PSNR: {val_psnr:.2f} dB")
        
        scheduler.step()
        
        # Save checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path = os.path.join(args.output_dir, 'liftholo_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'config': config,
            }, checkpoint_path)
            print(f"Saved best checkpoint to {checkpoint_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'liftholo_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'config': config,
            }, checkpoint_path)
    
    print("\nTraining complete!")
    print(f"Best checkpoint saved to: {os.path.join(args.output_dir, 'liftholo_best.pth')}")


if __name__ == '__main__':
    main()
