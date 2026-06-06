import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2

class SingleAgentWrapper(gym.Env):
    def __init__(self, world, scenario, add_coord_noise=False):
        super().__init__()
        self.world = world
        self.scenario = scenario
        self.world.discrete_action = False
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # 17-d：pos(2), vel(2), rel_sh(2), rel_pr(2), pr_vel(2), rel_lands(6), time(1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        
        self.steps = 0
        self.max_steps = 1000
        self.local_total_steps = 0 
        self.difficulty_override = None

        self.p_vel_ema = np.zeros(2)
        self.predator_speed_min = 0.4
        self.predator_speed_max = 1.01

        self.add_coord_noise = add_coord_noise

    def set_difficulty(self, progress):
        self.difficulty_override = progress

    def _add_sensor_noise(self, state, noise_level=0.03):
        noise = np.random.normal(0, noise_level, size=state.shape)
        noise[16] = 0
        return state + noise

    def _get_gt_state(self):
        p, pred = self.world.agents[1], self.world.agents[0]
        pos, vel = p.state.p_pos, p.state.p_vel
        rel_sh = self.world.check[0].state.p_pos - pos
        rel_pr = pred.state.p_pos - pos 
        pr_vel = pred.state.p_vel
        rel_lands = [l.state.p_pos - pos for l in self.world.landmarks]
        time_left = [(self.max_steps - self.steps) / self.max_steps]
        
        state =  np.concatenate([
            pos, vel, rel_sh, rel_pr, pr_vel, 
            np.concatenate(rel_lands), time_left
        ]).astype(np.float32)

        if self.add_coord_noise: 
            return self._add_sensor_noise(state.astype(np.float32))

        return state.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # fixed landmarks
        self.scenario.reset_world(self.world)

        self.steps = 0
        
        landmarks = self.world.landmarks

        p_agent = self.world.agents[1]
        while True:
            pos = np.random.uniform(-0.6, 0.6, 2)
            collision = any([np.linalg.norm(pos - l.state.p_pos) < (p_agent.size + l.size) for l in landmarks])

            if not collision:
                p_agent.state.p_pos = pos
                break

        check_pt = self.world.check[0]
        while True:
            target_pos = np.random.uniform(-0.7, 0.7, 2)
            dist_to_prey = np.linalg.norm(target_pos - p_agent.state.p_pos)
            collision = any([np.linalg.norm(target_pos - l.state.p_pos) < (check_pt.size + l.size) for l in landmarks])

            if dist_to_prey > 0.3 and not collision:
                check_pt.state.p_pos = target_pos
                break

        predator = self.world.agents[0]
        while True:
            pr_pos = np.random.uniform(-0.8, 0.8, 2)
            dist_to_prey = np.linalg.norm(pr_pos - p_agent.state.p_pos)
            collision = any([np.linalg.norm(pr_pos - l.state.p_pos) < (predator.size + l.size) for l in landmarks])
            
            if dist_to_prey > 0.06 and not collision:
                predator.state.p_pos = pr_pos
                break

        for a in self.world.agents: 
            a.state.p_vel = np.zeros(2)
        self.p_vel_ema = np.zeros(2)
        
        return self._get_gt_state(), {}

    def render(self, mode="rgb_array"):
        canvas = np.ones((800, 800, 3), dtype=np.uint8) * 255
        
        def to_pixel(pos):
            x_pix = int((pos[0] + 1.0) / 2.0 * 800)
            y_pix = int((1.0 - pos[1]) / 2.0 * 800)
            return (x_pix, y_pix)
        
        for entity in self.world.entities:
            pos = to_pixel(entity.state.p_pos)
            color_rgb = (int(entity.color[0] * 255), int(entity.color[1] * 255), int(entity.color[2] * 255))
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            r = int(entity.size / 2.0 * 800)
            
            if 'border' in entity.name or 'check' in entity.name:
                cv2.rectangle(canvas, (pos[0]-r, pos[1]-r), (pos[0]+r, pos[1]+r), color_bgr, -1)
            else:
                cv2.circle(canvas, pos, r, color_bgr, -1)

        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        if mode == "rgb_array":
            return [img_rgb]
        return None

    def step(self, action):
        p, pred = self.world.agents[1], self.world.agents[0]
        self.steps += 1
        self.local_total_steps += 1

        if self.difficulty_override is not None:
            progress = self.difficulty_override
        else:
            progress = 0.0
        current_speed = self.predator_speed_min + (self.predator_speed_max - self.predator_speed_min) * progress

        # PN
        current_lead = 0.4 * progress
        
        self.p_vel_ema = 0.8 * self.p_vel_ema + 0.2 * p.state.p_vel
        target_pos = p.state.p_pos + self.p_vel_ema * current_lead

        vec_p = target_pos - pred.state.p_pos
        pred.action.u = (vec_p / (np.linalg.norm(vec_p) + 1e-5)) * current_speed
        
        p.action.u = action
        self.world.step()
        
        obs = self._get_gt_state()
        reward = -0.05
        
        dist_sh = np.linalg.norm(p.state.p_pos - self.world.check[0].state.p_pos)
        success = dist_sh < (p.size + self.world.check[0].size)
        caught = self.scenario.is_collision(p, pred)

        hit_landmark = any([self.scenario.is_collision(p, l) for l in self.world.landmarks])
        hit_border = any([self.scenario.is_collision(p, b) for b in self.world.borders])

        hit_bad = hit_landmark or hit_border or np.any(np.abs(p.state.p_pos) > 0.99)

        terminated = False
        truncated = False
        outcome = "playing"

        if success:
            reward += 200.0; terminated = True; outcome = "WIN"
        elif caught:
            reward -= 100.0; terminated = True; outcome = "CAUGHT"
        elif hit_bad:
            reward -= 150.0; terminated = True; outcome = "SUICIDE"
        elif self.steps >= self.max_steps:
            reward -= 120.0; truncated = True; outcome = "TIMEOUT"
            
        return obs, float(reward), terminated, truncated, {"outcome": outcome}


class ImageObservationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=(3, 128, 128), dtype=np.uint8),
            "time_left": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._convert_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        info["gt_obs"] = obs 
        
        return self._convert_obs(obs), reward, terminated, truncated, info

    def _convert_obs(self, obs):
        img_800 = self.env.render("rgb_array")[0]
        img_128 = cv2.resize(img_800, (128, 128))
        img_transposed = np.transpose(img_128, (2, 0, 1))
        
        time_left = np.array([(self.env.max_steps - self.env.steps) / self.env.max_steps], dtype=np.float32)
        
        return {
            "image": img_transposed,
            "time_left": time_left
        }