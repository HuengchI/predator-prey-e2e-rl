import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
from torch.utils.data import random_split
import torchvision.transforms as T
from pathlib import Path

class DSNT(nn.Module):
    def __init__(self, height, width):
        super().__init__()

        x_range = torch.linspace(-1.0, 1.0, width)
        y_range = torch.linspace(-1.0, 1.0, height)
        grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
        self.register_buffer('grid_x', grid_x.clone().contiguous())
        self.register_buffer('grid_y', grid_y.clone().contiguous())

    def forward(self, heatmaps):
        B, N, H, W = heatmaps.shape
        heatmaps = heatmaps.view(B, N, -1)
        # Temperature scaling
        probs = F.softmax(heatmaps * 50, dim=-1)
        probs = probs.view(B, N, H, W)

        expected_x = torch.sum(probs * self.grid_x, dim=[2, 3])
        expected_y = torch.sum(probs * self.grid_y, dim=[2, 3])
        coords = torch.stack([expected_x, expected_y], dim=2).view(B, N * 2)

        var_x = torch.sum(probs * (self.grid_x.unsqueeze(0).unsqueeze(0) - expected_x.unsqueeze(-1).unsqueeze(-1)) ** 2, dim=[2, 3])
        var_y = torch.sum(probs * (self.grid_y.unsqueeze(0).unsqueeze(0) - expected_y.unsqueeze(-1).unsqueeze(-1)) ** 2, dim=[2, 3])

        loss_var = torch.mean(var_x + var_y)

        return coords, loss_var

class VisionEncoder(nn.Module):
    def __init__(self, num_entities=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(), # 64x64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(128, num_entities, kernel_size=1)
        )
        self.dsnt = DSNT(64, 64)

    def forward(self, x):
        heatmaps = self.features(x)
        return self.dsnt(heatmaps)

class MPEVisionDataset(Dataset):
    def __init__(self, img_path, coord_path, augmentation=False):
        self.img_path = img_path
        self.coord_path = coord_path

        _temp_imgs = np.load(img_path, mmap_mode='r')
        self.length = len(_temp_imgs)
        
        self.images = None
        self.coords = None

        self.augmentation = augmentation
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

    def __len__(self):
        return self.length

    def init_worker(self):
        self.images = np.load(self.img_path, mmap_mode='r')

        if self.augmentation:
            img = self.color_jitter(img)
            if torch.rand(1).item() > 0.7:
                noise = torch.randn_like(img) * 0.02
                img = torch.clamp(img + noise, 0.0, 1.0)

        self.coords = np.load(self.coord_path, mmap_mode='r')

    def __getitem__(self, idx):
        if self.images is None:
            self.init_worker()
            
        img_raw = self.images[idx]
        coord_raw = self.coords[idx]
        
        img = torch.from_numpy(img_raw.copy()).permute(2, 0, 1).float() / 255.0
        coord = torch.from_numpy(coord_raw.copy()).float()
        
        return img, coord


def compute_pixel_errors(pred, target, height=128, width=128):
    b, c = pred.shape
    pred_coords = pred.view(b, -1, 2)
    target_coords = target.view(b, -1, 2)
    
    scale = torch.tensor([(width - 1) / 2.0, (height - 1) / 2.0], device=pred.device)

    pred_pixel = (pred_coords + 1.0) * scale
    target_pixel = (target_coords + 1.0) * scale

    distances = torch.norm(pred_pixel - target_pixel, p=2, dim=-1)
    
    mean_err = torch.mean(distances).item()
    max_err = torch.max(distances).item()
    
    return mean_err, max_err

def seed_worker(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        dataset.init_worker()


def train_vision_model():
    epochs = 15

    run_id_str = f"vision/{time.strftime(r'%Y_%m_%d/%H_%M_%S', time.localtime())}"
    vision_nn_ckpt = Path(f"./outputs/").absolute() / run_id_str
    os.makedirs(vision_nn_ckpt, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Vision Backbone is now training on {device} ...")

    dataset = MPEVisionDataset("./outputs/vision_dataset/2026_06_01/09_34_20/images.npy", "./outputs/vision_dataset/2026_06_01/09_34_20/coords.npy")

    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=256, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=True,
    )

    model = VisionEncoder().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_steps,
        eta_min=1e-5
    )


    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        train_epoch_mean_err = 0
        
        for i, (batch_img, batch_coord) in enumerate(train_loader):
            batch_img, batch_coord = batch_img.to(device), batch_coord.to(device)
            
            pred_coord, loss_var = model(batch_img)
            loss_mse = F.mse_loss(pred_coord, batch_coord)
            loss = loss_mse + 0.01 * loss_var
            
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            scheduler.step()
            
            total_loss += loss.item()
            

            batch_mean_err, batch_max_err = compute_pixel_errors(pred_coord, batch_coord, height=64, width=64)
            train_epoch_mean_err += batch_mean_err

            if i % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(train_loader)}] "
                      f"Loss: {loss_mse.item():.6f} | "
                      f"Batch Mean Err: {batch_mean_err:.2f}px, Max Err: {batch_max_err:.2f}px | "
                      f"LR: {current_lr:.2e}")


        avg_loss = total_loss / len(train_loader)
        avg_train_pixel_err = train_epoch_mean_err / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"--- Epoch {epoch+1} Avg Loss: {avg_loss:.8f} | Train Mean Pixel Err: {avg_train_pixel_err:.2f}px | LR: {current_lr:.2e} ---")
        
        # Periodically save
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(vision_nn_ckpt, f"vision_encoder_ep{epoch+1}.pth"))

        model.eval()
        val_loss = 0
        val_epoch_mean_err = 0
        val_global_max_err = 0
        with torch.no_grad():
            for batch_img, batch_coord in val_loader:
                batch_img, batch_coord = batch_img.to(device), batch_coord.to(device)
                pred_coord, loss_var = model(batch_img)
                loss = F.mse_loss(pred_coord, batch_coord) + 0.01 * loss_var
                val_loss += loss.item()

                b_mean_err, b_max_err = compute_pixel_errors(pred_coord, batch_coord, height=64, width=64)
                val_epoch_mean_err += b_mean_err
                if b_max_err > val_global_max_err:
                    val_global_max_err = b_max_err
        
        val_loss /= len(val_loader)
        avg_val_pixel_err = val_epoch_mean_err / len(val_loader)
        print(f"--- Epoch {epoch+1} | Train Loss: {avg_loss:.8f} (Pixel Err: {avg_train_pixel_err:.2f}px) | "
              f"Val Loss: {val_loss:.8f} (Val Mean Err: {avg_val_pixel_err:.2f}px, Val Max Err: {val_global_max_err:.2f}px) ---")

        # Model selection
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(vision_nn_ckpt, "best_model.pth"))
            print(f">>> New Best Model Saved: Val Loss: {best_val_loss:.8f} | Pixel Err: {avg_val_pixel_err:.2f}px")

    save_path = os.path.join(vision_nn_ckpt, "vision_encoder_final.pth")
    torch.save(model.state_dict(), save_path)
    print(f">>> Vision Encoder training completed, final checkpoint saved at {save_path}")

if __name__ == "__main__":
    train_vision_model()
