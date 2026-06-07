import sys
import importlib
import os
import time
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback

DYNAMIC_LIB_ROOT = Path("./multiagent-envs").absolute()
sys.path.append(str(DYNAMIC_LIB_ROOT))

from src.env.env_wrapper import SingleAgentWrapper, ImageObservationWrapper
from src.vision.train_vision import VisionEncoder

def custom_load_scenario(scenario_name):
    path = DYNAMIC_LIB_ROOT / "multiagent" / "scenarios" / scenario_name
    if not os.path.exists(path):
        raise FileNotFoundError(f"Unable to localte the scenario file: {path}")

    spec = importlib.util.spec_from_file_location("custom_scenario", str(path))
    scenario_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scenario_module)
    return scenario_module.Scenario()

def make_env(rank, seed=0):
    def _init():
        set_random_seed(seed + rank)
        scenario_real = custom_load_scenario("simple_tag.py") 
        world = scenario_real.make_world()
        env = SingleAgentWrapper(world=world, scenario=scenario_real)
        env.set_difficulty(1.0)
        env = ImageObservationWrapper(env)
        return env
    return _init

def make_eval_env(rank, seed=0):
    def _init():
        set_random_seed(seed + rank)

        path = DYNAMIC_LIB_ROOT / "multiagent" / "scenarios" / "simple_tag.py"
        spec = importlib.util.spec_from_file_location("custom_scenario", str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        env = SingleAgentWrapper(mod.Scenario().make_world(), mod.Scenario())
        env.set_difficulty(1.0)
        env = ImageObservationWrapper(env)

        env = EvalSuccessRewardWrapper(env)
        return env
    return _init

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

class VisionMonitorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        obs_dict = self.locals['new_obs']
        
        with torch.no_grad():
            imgs = torch.tensor(obs_dict["image"]).float().to(self.model.device)
            if imgs.max() > 1.0:
                imgs = imgs / 255.0
            
            img_t = imgs[:, 3:6, :, :] # current frame
            
            pred_coords, _ = self.model.policy.actor.features_extractor.vision(img_t)

            pred_coords_phys = pred_coords.view(-1, 6, 2).clone()
            pred_coords_phys[:, :, 1] = -pred_coords_phys[:, :, 1] # correct y-axis sign
            pred_coords_phys = pred_coords_phys.view(-1, 12)
        
        # Env 17D Observation
        # 0-1: prey_pos, 2-3: vel, 4-5: rel_shelter, 6-7: rel_predator, 8-9: pr_vel, 10-15: rel_lands, 16: time
        gt_obs_list = [info.get("gt_obs") for info in self.locals['infos']]
        if gt_obs_list[0] is not None:
            # [B,17]
            gt_obs_tensor = torch.tensor(np.array(gt_obs_list), device=self.model.device).float()

            gt_prey_pos = gt_obs_tensor[:, 0:2]
            gt_check_pos = gt_obs_tensor[:, 4:6] + gt_prey_pos
            gt_pred_pos = gt_obs_tensor[:, 6:8] + gt_prey_pos
            gt_landmarks_pos = gt_obs_tensor[:, 10:16] + gt_prey_pos.repeat(1, 3)
            gt_coords_12d = torch.cat([
                gt_prey_pos, gt_pred_pos, gt_check_pos, gt_landmarks_pos
            ], dim=1)

            mean_pixel_err, max_pixel_err = compute_pixel_errors(pred_coords_phys, gt_coords_12d)

            self.logger.record("vision/mean_pixel_error", mean_pixel_err)
            self.logger.record("vision/max_pixel_error", max_pixel_err)

        return True

class E2EVisionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 17):
        super().__init__(observation_space, features_dim)
        self.vision = VisionEncoder(num_entities=6)
        self.env_dt = 0.1

    def forward(self, observations):
        imgs = observations["image"].float()
        if imgs.max() > 1.0:
            imgs = imgs / 255.0
        
        # Decouple last frame (t-1) and current frame (t)
        img_t_minus_1 = imgs[:, 0:3, :, :]
        img_t = imgs[:, 3:6, :, :]

        coords_t_minus_1, _ = self.vision(img_t_minus_1)
        coords_t, _ = self.vision(img_t)

        coords_t_phys = coords_t.view(-1, 6, 2).clone()
        coords_t_phys[:, :, 1] = -coords_t_phys[:, :, 1] # Y_physical = -Y_image
        coords_t_phys = coords_t_phys.view(-1, 12)
        
        coords_t_minus_1_phys = coords_t_minus_1.view(-1, 6, 2).clone()
        coords_t_minus_1_phys[:, :, 1] = -coords_t_minus_1_phys[:, :, 1]
        coords_t_minus_1_phys = coords_t_minus_1_phys.view(-1, 12).detach() # cutoff grad path
        
        # prey_pos, predator_pos, shelter_pos, land1_pos, land2_pos, land3_pos
        prey_pos = coords_t_phys[:, 0:2]
        pred_pos = coords_t_phys[:, 2:4]
        check_pos = coords_t_phys[:, 4:6]
        landmarks_pos = coords_t_phys[:, 6:12]
        
        rel_sh = check_pos - prey_pos
        rel_pr = pred_pos - prey_pos
        rel_lands = landmarks_pos - prey_pos.repeat(1, 3)
        
        # NOTE: 1D forward finite difference exhibits 1st-order boundary error
        prey_vel = (prey_pos - coords_t_minus_1[:, 0:2]) / self.env_dt
        pred_vel = (pred_pos - coords_t_minus_1[:, 2:4]) / self.env_dt

        time_left = observations["time_left"][:, -1:]

        # 17-D obs
        # 0-1: pos, 2-3: vel, 4-5: rel_sh, 6-7: rel_pr, 8-9: pr_vel, 10-15: rel_lands, 16: time
        obs_17d = torch.cat([
            prey_pos, prey_vel, rel_sh, rel_pr, pred_vel, rel_lands, time_left
        ], dim=1)

        return obs_17d


def create_model(env, vision_ckpt_path, pretrained_sac_path, log_path, device="cuda"):
    """
    Build an E2E model and load pretrained weights from two stems.
    """
    policy_kwargs = dict(
        features_extractor_class=E2EVisionExtractor,
        features_extractor_kwargs=dict(features_dim=17),
        net_arch=[512, 512],
        share_features_extractor=True
    )

    # Instantialize SAC with randomly initialized weights
    # Use a placeholder lr here, will be rewritten later
    model = SAC("MultiInputPolicy", 
                env,
                policy_kwargs=policy_kwargs,
                buffer_size=100000,
                batch_size=256,
                device=device,
                tensorboard_log=log_path,
                verbose=1)

    # Load pretrained weights
    print(f">>> Loading vision backbone weights: {vision_ckpt_path}")
    vision_weights = torch.load(vision_ckpt_path, map_location=device)
    # As the actor/critic use shared parameter, just override either one of them
    model.policy.critic.features_extractor.vision.load_state_dict(vision_weights)
    # Critic target should not share the features extractor with critic
    model.policy.critic_target.features_extractor.vision.load_state_dict(vision_weights)

    print(f">>> Loading pretrained policy weights: {pretrained_sac_path}")
    pretrained_sac = SAC.load(pretrained_sac_path, device="cpu")

    # Actor network
    # latent_pi, mu, log_std
    policy_linear_keys = ['latent_pi', 'mu', 'log_std']
    for key in policy_linear_keys:
        old_module = getattr(pretrained_sac.policy.actor, key)
        new_module = getattr(model.policy.actor, key)
        new_module.load_state_dict(old_module.state_dict())

    # Critic and Critic_target networks
    # qf0, qf1
    for i in range(len(model.policy.critic.q_networks)):
        model.policy.critic.q_networks[i].load_state_dict(
            pretrained_sac.policy.critic.q_networks[i].state_dict()
        )
        model.policy.critic_target.q_networks[i].load_state_dict(
            pretrained_sac.policy.critic_target.q_networks[i].state_dict()
        )

    # Critic_target should not be updated
    model.policy.critic_target.requires_grad_(False)
    model.policy.critic_target.eval()
    # Put a small sanity check here
    assert id(model.policy.actor.features_extractor) == id(model.policy.critic.features_extractor) 
    assert id(model.policy.actor.features_extractor) != id(model.policy.critic_target.features_extractor)
    
    print(">>> [Model] End2End model is ready.")
    return model


def setup_e2e_optimizers(model, mlp_lr=3e-4, vision_lr=1e-5, freeze_vision=True):
    """
    Configure optimizers for the end-to-end SAC agent with separate learning rates for MLP and vision backbone.
    
    Constructs parameter groups for actor and critic optimizers:
    - Actor optimizer: only updates the MLP components with `mlp_lr`.
    - Critic optimizer: updates MLP components and optionally the vision backbone with different LRs.
    - Vision backbone can be frozen to disable training.
    
    Args:
        model: SAC model containing actor and critic networks.
        mlp_lr: Learning rate for the MLP layers (actor/critic).
        vision_lr: Learning rate for the vision backbone (used only if not frozen).
        freeze_vision: If True, freeze the vision backbone and exclude it from optimization.
    """
    ext = model.policy.actor.features_extractor

    # maybe freeze vision backbone
    for param in ext.vision.parameters():
        param.requires_grad = not freeze_vision

    # Filter in vision network parameter keys
    vision_params = list(ext.vision.parameters())
    vision_param_ids = set(id(p) for p in vision_params)

    # Actor optimizer
    actor_mlp_params = [p for p in model.actor.parameters() if id(p) not in vision_param_ids]
    actor_param_groups = [
        {'params': actor_mlp_params, 'lr': mlp_lr}
    ]
    model.actor.optimizer = torch.optim.Adam(actor_param_groups)

    # Critic optimizer
    critic_mlp_params = [p for p in model.critic.parameters() if id(p) not in vision_param_ids]
    critic_param_groups = [
        {'params': critic_mlp_params, 'lr': mlp_lr}
    ]
    if not freeze_vision:
        critic_param_groups.append({'params': vision_params, 'lr': vision_lr})

    model.critic.optimizer = torch.optim.Adam(critic_param_groups)

    print(f">>> [Optimizer] Optimizer is ready. Current configuration: Freeze Vision = {freeze_vision}")
    print(f">>> Learning rate of SAC MLP: {mlp_lr:.2e} | Learning rate of vision backbone: {(vision_lr if not freeze_vision else 0.0):.2e}")


def main():

    run_id_str = f"e2e/{time.strftime(r'%Y_%m_%d/%H_%M_%S', time.localtime())}"
    tb_log_path = Path("./logs/").absolute() / run_id_str
    output_base = Path(f"./outputs/").absolute() / run_id_str

    eval_save_path = output_base
    model_save_path = output_base / 'final_model'
    progress_save_path = output_base / 'progress_checkpoints/'

    progress_save_path.mkdir(parents=True, exist_ok=True)
    tb_log_path.mkdir(parents=True, exist_ok=True)


    num_cpu = 64
    num_cpu_eval = 32
    total_timesteps = 30_000_000
    success_callback_window = 500
    eval_epoches=200
    eval_freq = 100_000
    save_freq = 400_000
    # End2End Training
    freeze_vision_backbone = False


    # Construct Envs
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=2)

    eval_env = VecMonitor(SubprocVecEnv([make_eval_env(i, seed=937) for i in range(num_cpu_eval)]))
    eval_env = VecFrameStack(eval_env, n_stack=2)

    # Construct models
    model = create_model(
        env=env,
        vision_ckpt_path="./outputs/vision/2026_06_01/19_53_14/best_model.pth",
        pretrained_sac_path="./outputs/oracle/2026_06_01/11_15_29/best_model.zip",
        device="cuda",
        log_path=tb_log_path
    )

    # Override optimizers
    setup_e2e_optimizers(
        model=model, 
        mlp_lr=3e-4, 
        vision_lr=1e-6, 
        freeze_vision=freeze_vision_backbone
    )

    # Callbacks
    success_callback = SuccessRateCallback(window_size=success_callback_window)
    eval_callback = EvalCallback(eval_env, best_model_save_path=eval_save_path, n_eval_episodes=eval_epoches, eval_freq=eval_freq // num_cpu, deterministic=True)
    checkpoint_callback = CheckpointCallback(save_freq=save_freq // num_cpu, save_path=progress_save_path, name_prefix='sac_e2e')
    vision_monitor_callback = VisionMonitorCallback(verbose=1)
    
    # Start training
    print(">>> Start E2E training now ...")
    try:
        model.learn(total_timesteps=total_timesteps,
                    progress_bar=True,
                    log_interval=1,
                    callback=[success_callback, vision_monitor_callback, checkpoint_callback, eval_callback, ]
                )
    except KeyboardInterrupt:
        print("Ctrl+C detected. Saving last model checkpoint...")
    finally:
        model.save(model_save_path)
        print(f"Final model saved at {model_save_path}.zip")

if __name__ == "__main__":
    main()

