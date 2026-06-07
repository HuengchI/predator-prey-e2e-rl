import os
import sys
import time
import numpy as np
import cv2
import importlib.util
from pathlib import Path
from multiprocessing import Pool
import hashlib
from tqdm import tqdm

DYNAMIC_LIB_ROOT = Path("./multiagent-envs").absolute()
sys.path.append(str(DYNAMIC_LIB_ROOT))
from src.env.env_wrapper import SingleAgentWrapper 

def render_by_opencv(world):
    canvas = np.ones((800, 800, 3), dtype=np.uint8) * 255
    def to_pixel(pos):
        x_pix = int((pos[0] + 1.0) / 2.0 * 800)
        y_pix = int((1.0 - pos[1]) / 2.0 * 800)
        return (x_pix, y_pix)
    
    for entity in world.entities:
        pos = to_pixel(entity.state.p_pos)
        color = (int(entity.color[2] * 255), int(entity.color[1] * 255), int(entity.color[0] * 255))
        r = int(entity.size / 2.0 * 800)
        if 'border' in entity.name or 'check' in entity.name:
            cv2.rectangle(canvas, (pos[0]-r, pos[1]-r), (pos[0]+r, pos[1]+r), color, -1)
        else:
            cv2.circle(canvas, pos, r, color, -1)
    return canvas


def worker_fn(payload):
    rank, num_samples = payload

    np.random.seed(os.getpid() + rank)
    cv2.setNumThreads(0) 
    
    path = DYNAMIC_LIB_ROOT / "multiagent" / "scenarios" / "simple_tag.py"
    spec = importlib.util.spec_from_file_location("mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    scenario = mod.Scenario()
    world = scenario.make_world()
    env = SingleAgentWrapper(world, scenario)

    env.set_difficulty(1.0)
    
    local_images = []
    local_coords = []

    for i in range(num_samples):
        # unique seed hash
        seed_string = f"{os.getpid()}_{rank}_{i}"
        hash_seed = int(hashlib.sha256(seed_string.encode()).hexdigest(), 16) % (2**32)
        obs, _ = env.reset(seed=hash_seed)

        img_raw = render_by_opencv(env.world)
        img_resized = cv2.resize(img_rgb := cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB), (128, 128))
        
        # GT position
        my_pos = obs[0:2]
        sh_pos = my_pos + obs[4:6]
        pr_pos = my_pos + obs[6:8]
        l1_pos = my_pos + obs[10:12]
        l2_pos = my_pos + obs[12:14]
        l3_pos = my_pos + obs[14:16]

        # X_img = X_phy, Y_img = -Y_phy
        def to_img_gt(pos):
            return np.array([pos[0], -pos[1]], dtype=np.float32)

        local_coords.append(np.concatenate([
            to_img_gt(my_pos), 
            to_img_gt(pr_pos), 
            to_img_gt(sh_pos), 
            to_img_gt(l1_pos),
            to_img_gt(l2_pos),
            to_img_gt(l3_pos)
        ]))
        
        local_images.append(img_resized)
        
    return np.array(local_images), np.array(local_coords)

def parallel_collect(total_samples=50_000, num_workers=16):
    run_id_str = f"vision_dataset/{time.strftime(r'%Y_%m_%d/%H_%M_%S', time.localtime())}"

    save_dir = Path("./outputs/").absolute() / run_id_str
    save_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = [(i, total_samples // num_workers) for i in range(num_workers)]
    
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(worker_fn, tasks), total=num_workers))
    
    imgs = np.concatenate([r[0] for r in results])
    coords = np.concatenate([r[1] for r in results])
    
    np.save(save_dir / "images.npy", imgs)
    np.save(save_dir / "coords.npy", coords)
    print(f"Data collected: {imgs.shape} samples saved to {save_dir}")

if __name__ == "__main__":
    parallel_collect(total_samples=300_000, num_workers=64)
