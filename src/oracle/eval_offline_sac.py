import os
import sys
import glob
import time
import numpy as np
from pathlib import Path
import importlib.util
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import SAC
import gymnasium as gym


DYNAMIC_LIB_ROOT = Path("./multiagent-envs").absolute()
sys.path.append(str(DYNAMIC_LIB_ROOT))
from src.env.env_wrapper import SingleAgentWrapper 

def make_eval_env(rank, seed=0):
    def _init():
        # RNG seed
        set_random_seed(seed + rank)
        path = DYNAMIC_LIB_ROOT / "multiagent" / "scenarios" / "simple_tag.py"
        spec = importlib.util.spec_from_file_location("custom_scenario", str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        env = SingleAgentWrapper(mod.Scenario().make_world(), mod.Scenario())

        env.set_difficulty(1.0) 
        return env

    return _init


def get_sorted_checkpoints(progress_checkpoint_dir, extra_checkpoint_paths):
    files = glob.glob(os.path.join(progress_checkpoint_dir, "*.zip"))
    
    def extract_steps(filepath):
        basename = os.path.basename(filepath)
        parts = basename.split('_')
        for part in parts:
            if part.isdigit():
                return int(part)
        return 0

    # steps ascending order
    sorted_files = sorted(files, key=extract_steps)

    for p in extra_checkpoint_paths:
        if os.path.exists(p):
            sorted_files.append(p)
        
    return sorted_files


def parallel_evaluate_model(model_path, eval_env, num_eval_envs, target_episodes=1000):
    print(f"\n[{os.path.basename(model_path)}] in parallel evaluation...")
    
    try:
        model = SAC.load(model_path)
    except Exception as e:
        print(f"[Error] Can't load model {model_path}: {e}")
        return

    stats = {"WIN": 0, "CAUGHT": 0, "SUICIDE": 0, "TIMEOUT": 0}
    win_steps = []
    current_steps = np.zeros(num_eval_envs, dtype=int)
    
    initial_states_fingerprints = set()
    started_episodes = 0
    completed_episodes = 0
    
    obs = eval_env.reset()
    
    for i in range(num_eval_envs):
        if started_episodes < target_episodes:
            initial_states_fingerprints.add(obs[i].tobytes())
            started_episodes += 1

    start_time = time.time()

    while completed_episodes < target_episodes:
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(actions)
        current_steps += 1
        
        for i, done in enumerate(dones):
            if done:
                outcome = infos[i].get("outcome", "TIMEOUT")

                if completed_episodes < target_episodes:
                    stats[outcome] = stats.get(outcome, 0) + 1
                    if outcome == "WIN":
                        win_steps.append(current_steps[i])
                    completed_episodes += 1
                    
                    if started_episodes < target_episodes:
                        initial_states_fingerprints.add(obs[i].tobytes())
                        started_episodes += 1
                
                current_steps[i] = 0

    elapsed_time = time.time() - start_time

    win_rate = (stats['WIN'] / target_episodes) * 100
    caught_rate = (stats['CAUGHT'] / target_episodes) * 100
    suicide_rate = (stats['SUICIDE'] / target_episodes) * 100
    timeout_rate = (stats['TIMEOUT'] / target_episodes) * 100
    avg_win_steps = np.mean(win_steps) if win_steps else 0.0
    
    unique_starts = len(initial_states_fingerprints)

    print(f"  --> Time Used: {elapsed_time:.2f} s | Success Rate: \033[92m{win_rate:5.1f}%\033[0m | Efficiency: {avg_win_steps:.1f} step")
    print(f"  --> Distribution | CAUGHT: {caught_rate:4.1f}% | SUICIDE: {suicide_rate:4.1f}% | TIMEOUT: {timeout_rate:4.1f}%")
    
    if unique_starts == target_episodes:
        print(f"  --> \033[96m[RNG Verification Pass]\033[0m Evaluated on {unique_starts} different resets")
    else:
        print(f"  --> \033[91m[RNG Verification Warning]\033[0m {target_episodes - unique_starts} identical resets exist")

    return win_rate



def main():
    progress_checkpoint_dir = "./outputs/oracle/2026_06_01/09_15_29/progress_checkpoints"
    extra_checkpoint_paths = [
        "outputs/oracle/2026_06_01/09_15_29/best_model.zip",
        "outputs/oracle/2026_06_01/09_15_29/final_model.zip",
    ]

    num_eval_envs = 72
    target_episodes = 1000

    
    checkpoints = get_sorted_checkpoints(progress_checkpoint_dir, extra_checkpoint_paths)
    
    if not checkpoints:
        print("No checkpoints found. Check the path")
        return

    print(f"{len(checkpoints)} checkpoints found, launching {num_eval_envs} parallel evaluation...")
    
    eval_env = SubprocVecEnv([make_eval_env(i, seed=8888) for i in range(num_eval_envs)])
    
    best_ckpt = None
    best_win_rate = -1.0
    
    try:
        for ckpt in checkpoints:
            win_rate = parallel_evaluate_model(ckpt, eval_env, num_eval_envs, target_episodes)

            if win_rate is not None and win_rate > best_win_rate:
                best_win_rate = win_rate
                best_ckpt = ckpt
                
    except KeyboardInterrupt:
        print("\nCtrl+C Detected.")
    finally:
        eval_env.close()
        
    print(f"\n{'-'*50}")
    print(f"Evaluation Finished.")
    print(f"Best model: {os.path.basename(best_ckpt)} (Success Rate: {best_win_rate:.1f}%)")
    print(f"{'-'*50}")

if __name__ == "__main__":
    main()
