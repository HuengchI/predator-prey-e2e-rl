import os
import sys
from pathlib import Path
import cv2
import numpy as np
from stable_baselines3 import SAC
from env_wrapper import SingleAgentWrapper 
import importlib.util

DYNAMIC_LIB_ROOT = Path(__file__).absolute().parent.parent / "multiagent-envs-ML"
sys.path.append(str(DYNAMIC_LIB_ROOT))

def custom_load_scenario(scenario_name):
    path = DYNAMIC_LIB_ROOT / "multiagent" / "scenarios" / scenario_name
    if not os.path.exists(path):
        raise FileNotFoundError(f"Unable to find scenario: {path}")
    
    spec = importlib.util.spec_from_file_location("custom_scenario", str(path))
    scenario_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scenario_module)
    return scenario_module.Scenario()

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

def record_demo(model_path, video_name="prey_god_mode_sac.mp4", num_episodes=10):
    scenario = custom_load_scenario("simple_tag.py")
    world = scenario.make_world()

    env = SingleAgentWrapper(world, scenario)

    env.set_difficulty(1.0)

    if not os.path.exists(model_path):
        print(f"[Error] Can't find weights: {model_path}")
        return
        
    model = SAC.load(model_path)
    print(f"Successfully loaded SAC model from {model_path}")

    fourcc = cv2.VideoWriter_fourcc(*'vp09')
    out = cv2.VideoWriter(video_name, fourcc, 30.0, (800, 800))

    print(f"Recording {num_episodes} episodes to {video_name}...")

    success_count = 0

    for i in range(num_episodes):
        state, info = env.reset()
        done = False
        step_count = 0
        final_outcome = "TIMEOUT"
        
        while not done and step_count < 500:
            # SAC inference
            action, _ = model.predict(state, deterministic=True)
            
            state, reward, terminated, truncated, info = env.step(action)
            frame_bgr = render_by_opencv(env.world)
            
            cv2.putText(frame_bgr, f"Ep: {i+1} Step: {step_count}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            out.write(frame_bgr)
            
            done = terminated or truncated
            step_count += 1
            
            if terminated:
                final_outcome = info.get("outcome", "DONE")
            
            if done:
                if final_outcome == "WIN": success_count += 1
                # froze frame
                for _ in range(30): out.write(frame_bgr)

        if final_outcome == "WIN":
            color_code = "\033[92m" # Green
        elif final_outcome == "TIMEOUT":
            color_code = "\033[93m" # Yellow
        else:
            color_code = "\033[91m" # Red
            
        print(f"Episode {i+1:2d} | Steps: {step_count:3d} | Result: {color_code}{final_outcome}\033[0m")

    out.release()
    print(f"\nFinal Success Rate: {success_count/num_episodes * 100:.1f}%")
    print(f"Video saved as {video_name}")

if __name__ == "__main__":
    target_model = "./saved_models/sac_marathon_final.zip"

    record_demo(model_path=target_model)
