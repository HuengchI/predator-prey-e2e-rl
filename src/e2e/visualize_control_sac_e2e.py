import os
import sys
from pathlib import Path
import cv2
import numpy as np
from stable_baselines3 import SAC
import importlib.util
import math
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.env.env_wrapper import SingleAgentWrapper, ImageObservationWrapper
from src.e2e.train_e2e import E2EVisionExtractor, setup_e2e_optimizers

DYNAMIC_LIB_ROOT = Path("./multiagent-envs").absolute()
sys.path.append(str(DYNAMIC_LIB_ROOT))

def custom_load_scenario(scenario_name):
    path = DYNAMIC_LIB_ROOT / "multiagent" / "scenarios" / scenario_name
    if not os.path.exists(path):
        raise FileNotFoundError(f"Unbale to find scenario: {path}")
    
    spec = importlib.util.spec_from_file_location("custom_scenario", str(path))
    scenario_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scenario_module)
    return scenario_module.Scenario()

class CatchTerminalWorldWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            # mock_world
            class MockState:
                def __init__(self, pos): self.p_pos = pos.copy()
            class MockEntity:
                def __init__(self, name, size, color, pos):
                    self.name, self.size, self.color = name, size, color
                    self.state = MockState(pos)
            class MockWorld:
                def __init__(self, entities): self.entities = entities

            mock_entities = [MockEntity(e.name, e.size, e.color, e.state.p_pos) 
                             for e in self.env.unwrapped.world.entities]
            info['terminal_world'] = MockWorld(mock_entities)
        return obs, reward, terminated, truncated, info

def draw_dashed_circle(img, center, radius, color, thickness=2):
    pts = []

    for i in range(0, 360, 10):
        x = int(center[0] + radius * math.cos(math.radians(i)))
        y = int(center[1] + radius * math.sin(math.radians(i)))
        pts.append((x, y))

    for i in range(0, len(pts), 2):
        p1 = pts[i]
        p2 = pts[(i+1) % len(pts)]
        cv2.line(img, p1, p2, color, thickness)

def render_by_opencv(world):
    canvas = np.ones((800, 800, 3), dtype=np.uint8) * 255
    
    def to_pixel(pos):
        x_pix = int((pos[0] + 1.0) / 2.0 * 800)
        y_pix = int((1.0 - pos[1]) / 2.0 * 800)
        return (x_pix, y_pix)

    def to_pixel_size(size):
        return int(size / 2.0 * 800)

    for entity in world.entities:
        pos = to_pixel(entity.state.p_pos)
        color_bgr = (int(entity.color[2] * 255), int(entity.color[1] * 255), int(entity.color[0] * 255))
        
        if 'border' in entity.name or 'check' in entity.name:
            r_pix = to_pixel_size(entity.size)
            cv2.rectangle(canvas, (pos[0] - r_pix, pos[1] - r_pix), (pos[0] + r_pix, pos[1] + r_pix), color_bgr, -1)
        else:
            r_pix = to_pixel_size(entity.size)
            cv2.circle(canvas, pos, r_pix, color_bgr, -1)
            if 'agent' in entity.name:
                cv2.circle(canvas, pos, r_pix, (50, 50, 50), 2)

    return canvas

def record_demo(model_path, video_name, num_episodes=10):
    def _make_env():
        scenario = custom_load_scenario("simple_tag.py")
        world = scenario.make_world()
        raw_env = SingleAgentWrapper(world, scenario)
        raw_env.set_difficulty(1.0)
        img_env = ImageObservationWrapper(raw_env)
        return CatchTerminalWorldWrapper(img_env)

    env = DummyVecEnv([_make_env])
    env = VecFrameStack(env, n_stack=2)
    
    if not os.path.exists(model_path):
        print(f">>> [Err] Unable to find weights {model_path}")
        return
        
    print(f">>> Creating E2E model {model_path} ...")
    
    # Same as training time
    policy_kwargs = dict(
        features_extractor_class=E2EVisionExtractor,
        features_extractor_kwargs=dict(features_dim=17),
        net_arch=[512, 512],
        share_features_extractor=True
    )
    
    model = SAC("MultiInputPolicy", 
                env, 
                policy_kwargs=policy_kwargs, 
                device="cuda")
    
    # align optimizer setting up
    setup_e2e_optimizers(model, freeze_vision=False)
    
    model.set_parameters(model_path)
    
    model.policy.eval()
    print(">>> Model loaded.")
    # ==========================================

    fourcc = cv2.VideoWriter_fourcc(*'vp09')
    out = cv2.VideoWriter(video_name, fourcc, 30.0, (800, 800))

    slow_motion_factor = 1
    print(f">>> Recording {num_episodes} Episodes, output file: {video_name}...")

    success_count = 0

    for i in range(num_episodes):
        done = False
        step_count = 0
        final_outcome = "TIMEOUT"
        
        state = env.reset() 
        
        while not done and step_count < 500:
            # render real env
            underlying_world = env.envs[0].unwrapped.world
            frame_bgr = render_by_opencv(underlying_world)

            # render agent's view
            with torch.no_grad():
                imgs = torch.tensor(state["image"]).float().to(model.device)
                if imgs.max() > 1.0:
                    imgs = imgs / 255.0
                
                img_t = imgs[:, 3:6, :, :]
                pred_coords, _ = model.policy.actor.features_extractor.vision(img_t)
                pred_coords_phys = pred_coords.view(-1, 6, 2).cpu().numpy()[0]
                pred_coords_phys[:, 1] = -pred_coords_phys[:, 1]

            for j in range(6):
                pos = pred_coords_phys[j]
                x_pix = int((pos[0] + 1.0) / 2.0 * 800)
                y_pix = int((1.0 - pos[1]) / 2.0 * 800)
                draw_dashed_circle(frame_bgr, (x_pix, y_pix), radius=25, color=(255, 0, 255), thickness=2)

            cv2.putText(frame_bgr, "--- : AI Vision Prediction", (420, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

            # write video with slow motion
            for _ in range(slow_motion_factor):
                out.write(frame_bgr)

            # action step and env envolve
            action, _ = model.predict(state, deterministic=True)
            state, reward, done_arr, info = env.step(action)
            
            is_done = done_arr[0] 
            current_info = info[0]
            step_count += 1

            # closing
            if is_done:
                final_outcome = current_info.get("outcome", "DONE")
                if final_outcome == "WIN":
                    success_count += 1
                
                # frozen frame
                frozen_duration = 30*slow_motion_factor
                if 'terminal_world' in current_info:
                    final_world = current_info['terminal_world']

                    collision_frame = render_by_opencv(final_world)

                    cv2.putText(collision_frame, f"Ep: {i+1} Step: {step_count} ({final_outcome}!)", 
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    for _ in range(frozen_duration): 
                        out.write(collision_frame)
                else:
                    for _ in range(frozen_duration): 
                        out.write(frame_bgr)
                
                done = True

        if final_outcome == "WIN":
            color_code = "\033[92m" 
        elif final_outcome == "TIMEOUT":
            color_code = "\033[93m" 
        else:
            color_code = "\033[91m" 
            
        print(f"Episode {i+1:2d} | Steps: {step_count:3d} | Ending State: {color_code}{final_outcome}\033[0m")

    out.release()
    print(f"\n>>> Mean Success Rate: {success_count/num_episodes * 100:.1f}%")
    print(f">>> Video has saved to: {video_name}")

if __name__ == "__main__":
    target_model = "./outputs/e2e/2026_06_02/15_10_47/best_model.zip"
    video_name = "./outputs/e2e/2026_06_02/15_10_47/vid_e2e_sac_best_model.mp4"

    record_demo(model_path=target_model, video_name=video_name, num_episodes=10)
