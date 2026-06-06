import os
import sys
import numpy as np
from pathlib import Path
import importlib.util
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback
from stable_baselines3 import SAC
import gymnasium as gym

DYNAMIC_LIB_ROOT = Path(__file__).absolute().parent.parent / "multiagent-envs-ML"
sys.path.append(str(DYNAMIC_LIB_ROOT))
from env_wrapper import SingleAgentWrapper 

class EvalSuccessRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if 'outcome' in info and info['outcome'] != 'playing':
            reward = 1.0 if info['outcome'] == 'WIN' else 0.0
        else:
            reward = 0.0

        return obs, reward, terminated, truncated, info

def make_env(rank, seed=0):
    def _init():
        # RNG
        set_random_seed(seed + rank)

        path = DYNAMIC_LIB_ROOT / "multiagent" / "scenarios" / "simple_tag.py"
        spec = importlib.util.spec_from_file_location("custom_scenario", str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        env = SingleAgentWrapper(mod.Scenario().make_world(), mod.Scenario())
        return env
    return _init

def make_eval_env(rank, seed=0):
    def _init():
        # RNG
        set_random_seed(seed + rank)

        path = DYNAMIC_LIB_ROOT / "multiagent" / "scenarios" / "simple_tag.py"
        spec = importlib.util.spec_from_file_location("custom_scenario", str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        env = SingleAgentWrapper(mod.Scenario().make_world(), mod.Scenario())

        env.set_difficulty(1.0) 
        env = EvalSuccessRewardWrapper(env)
        return env
    return _init

class SuccessRateCallback(BaseCallback):
    def __init__(self, window_size=100, verbose=0):
        super().__init__(verbose)
        self.window_size = window_size
        self.successes = []

    def _on_step(self) -> bool:
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'outcome' in info and info['outcome'] != 'playing':
                    self.successes.append(1.0 if info['outcome'] == 'WIN' else 0.0)
        
        if len(self.successes) >= self.window_size:
            sr = np.mean(self.successes[-self.window_size:])
            self.logger.record("rollout/success_rate", sr)
        return True

class CurriculumCallback(BaseCallback):
    def __init__(self, eval_env, total_timesteps, warmup_ratio, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.total_timesteps = total_timesteps
        self.warmup_ratio = warmup_ratio

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / (self.total_timesteps * self.warmup_ratio), 1.0)
        self.training_env.env_method("set_difficulty", progress)
        
        self.logger.record("curriculum/progress", progress)
        return True

def main():
    num_cpu = 72
    num_cpu_eval = 50
    total_timesteps = int(3e7)
    curriculum_warmup_ratio = 0.33
    eval_epoches = 500
    eval_freq = 100_000
    save_freq = 1_000_000
    success_callback_window = 500

    train_freq=(1 * num_cpu, "step")
    gradient_steps=num_cpu

    eval_save_path = "./saved_models/overnight_sac_best/"
    progress_save_path = './saved_models/progress_checkpoints/'
    os.makedirs(eval_save_path, exist_ok=True)
    os.makedirs('./saved_models/progress_checkpoints/', exist_ok=True)
    
    train_env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(num_cpu)]))
    eval_env = VecMonitor(SubprocVecEnv([make_eval_env(i, seed=1337) for i in range(num_cpu_eval)]))

    success_callback = SuccessRateCallback(window_size=success_callback_window)

    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path=eval_save_path,
                                 n_eval_episodes=eval_epoches, 
                                 eval_freq=eval_freq // num_cpu, 
                                 deterministic=True)
    curriculum_callback = CurriculumCallback(eval_env=eval_env, total_timesteps=total_timesteps, warmup_ratio=curriculum_warmup_ratio)

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // num_cpu, 
        save_path=progress_save_path,
        name_prefix='sac_marathon'
    )

    model = SAC("MlpPolicy", 
                train_env, 
                learning_rate=3e-4, 
                buffer_size=total_timesteps,
                batch_size=1024,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                ent_coef=0.05,
                policy_kwargs=dict(net_arch=[512, 512]),
                verbose=1,
                tensorboard_log="./sac_marathon_logs/",
                learning_starts = 10000,
                device="cuda")

    print(f"SAC Oracle Training Starts... (CPU core: {num_cpu}, Total timesteps: {total_timesteps})")

    try:
        model.learn(total_timesteps=total_timesteps, 
                callback=[curriculum_callback, success_callback, checkpoint_callback, eval_callback], 
                progress_bar=True)
    except KeyboardInterrupt:
        print(f"Ctrl+C Signal Detected...")
    finally:
        model_save_path = "saved_models/sac_marathon_final"
        model.save(model_save_path)
        print(f"Checkpoint: {model_save_path}")

if __name__ == "__main__":
    main()
