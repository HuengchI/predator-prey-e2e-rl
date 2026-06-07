import numpy as np
from pathlib import Path
import hashlib

def run_macro_audit():
    data_dir = Path("./outputs/vision_dataset/2026_06_01/09_15_29/")
    img_path = data_dir / "images.npy"
    coord_path = data_dir / "coords.npy"
    
    if not img_path.exists() or not coord_path.exists():
        print(f"[Error] Unable to find dataset: {data_dir}")
        return

    print("Load datasets...")
    images = np.load(img_path)
    coords = np.load(coord_path)
    
    num_samples = len(images)
    print(f"Load successfully! Image shape: {images.shape} | Coordinate shape: {coords.shape}")
    print("-" * 60)

    # ==========================================
    # 1.Check for Clones
    # ==========================================
    print("Calculating byte hashing...")
    hashes = set()
    for img in images:
        hashes.add(hashlib.md5(img.tobytes()).hexdigest())
    
    unique_imgs = len(hashes)
    duplication_rate = (1.0 - unique_imgs / num_samples) * 100
    
    print(f"  --> Unique image numbers: {unique_imgs} / {num_samples}")
    if unique_imgs == num_samples:
        print("  --> \033[92m[PASS]\033[0m No identical image found!")
    else:
        print(f"  --> \033[91m[WARNING]\033[0m Duplication rate: {duplication_rate:.2f}%")

    # ==========================================
    # 2. Check for Domain Limits
    # ==========================================
    # Prey(0:2), Pred(2:4), Shelter(4:6), Land1(6:8)
    names = ["Prey", "Pred", "Shelter", "Land1"]
    
    print(f"  {'Entity Name':<12} | {'Min X':<7} | {'Max X':<7} | {'Min Y':<7} | {'Max Y':<7} | {'Mean':<10}")
    print("  " + "-" * 75)
    
    for i, name in enumerate(names):
        x_data = coords[:, i*2]
        y_data = coords[:, i*2 + 1]
        
        print(f"  {name:<12} | {x_data.min():.4f} | {x_data.max():.4f} | {y_data.min():.4f} | {y_data.max():.4f} | ({x_data.mean():.3f}, {y_data.mean():.3f})")
    
    if coords.min() < -1.0 or coords.max() > 1.0:
        print("\n  --> \033[91m[WARNING]\033[0m Coordinates exceed the [-1, 1] boundary, check normalization.")
    print("-" * 60)

if __name__ == "__main__":
    run_macro_audit()
