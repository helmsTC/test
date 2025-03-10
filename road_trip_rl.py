import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Offscreen rendering.
import matplotlib.pyplot as plt
import random
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fixed start and end coordinates.
denver = np.array([39.7392, -104.9903])
fort_collins = np.array([40.5853, -105.0844])

# --------------------------
# (Optional) Custom Torch Model for RLlib
# --------------------------
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2

class CustomConvModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # Expect input images of shape [B, H, W, C] (HWC); convert to CHW.
        # For an 84x84 input.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=8, stride=4, padding=0)  # [B,16,20,20]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0)  # [B,32,9,9]
        self.conv3 = nn.Conv2d(32, 8, kernel_size=9, stride=1, padding=0)   # [B,8,1,1]
        self._num_outputs = num_outputs
        self._features = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].permute(0, 3, 1, 2).float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  # Expected shape: [B,8,1,1]
        self._features = x.view(x.size(0), -1)  # [B,8]
        return x, state

    def value_function(self):
        return self._features.sum(1)  # Return [B]

# --------------------------
# Zone Generation Functions
# --------------------------
def create_traffic_zones(num_zones=0, **kwargs):
    # No traffic in this simplified version.
    return []

def create_accident_zones(num_zones=1, radius=0.04):
    # Fixed accident zone: use the midpoint between Denver and Fort Collins.
    center = (denver + fort_collins) / 2
    return [(center[0], center[1], radius)]

# --------------------------
# Helper Functions for Route Evaluation
# --------------------------
def compute_total_distance(waypoints):
    return sum(np.linalg.norm(waypoints[i+1]-waypoints[i]) for i in range(len(waypoints)-1))

def line_circle_intersection_length(p1, p2, C, r):
    """Compute the length of the segment from p1 to p2 that lies inside a circle with center C and radius r."""
    d = np.linalg.norm(p2 - p1)
    if d == 0:
        return 0.0
    # Represent the segment as P(t)=p1+t*(p2-p1), t in [0,1].
    # Solve ||P(t)-C||^2 = r^2, which is a quadratic in t.
    dp = p2 - p1
    A = np.dot(dp, dp)
    B = 2 * np.dot(dp, p1 - C)
    C_val = np.dot(p1 - C, p1 - C) - r**2
    discriminant = B**2 - 4*A*C_val
    if discriminant < 0:
        # No intersection. However, if both endpoints are inside, return full length.
        if np.linalg.norm(p1-C) < r and np.linalg.norm(p2-C) < r:
            return d
        return 0.0
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-B - sqrt_disc) / (2*A)
    t2 = (-B + sqrt_disc) / (2*A)
    # Clip to [0,1]
    t1 = np.clip(t1, 0, 1)
    t2 = np.clip(t2, 0, 1)
    return max(0, t2 - t1) * d

def total_intersection_length(waypoints, accident_zones):
    total = 0.0
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i+1]
        for az in accident_zones:
            C = np.array([az[0], az[1]])
            r = az[2]
            total += line_circle_intersection_length(p1, p2, C, r)
    return total
# --------------------------
# Cost Function and Optimization
# --------------------------
def route_cost(waypoints, accident_zones, coeffs):
    total_distance = compute_total_distance(waypoints)
    D0 = np.linalg.norm(fort_collins - denver)
    total_trip_time = 60 + (total_distance - D0)
    inter_length = total_intersection_length(waypoints, accident_zones)
    # Multiply intersection length by a high factor (e.g., 1000)
    intersection_penalty = 1000 * inter_length
    # Now cost is a weighted sum:
    cost = coeffs[1] * total_distance + coeffs[2] * total_trip_time + coeffs[0] * intersection_penalty
    return np.array(cost)

def optimize_route(start, end, accident_zones, coeffs, num_intermediate=3, iterations=300, step_size=0.01):
    coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
    waypoints = [start]
    for i in range(1, num_intermediate+1):
        alpha = i / (num_intermediate+1)
        waypoint = start + alpha*(end - start)
        waypoints.append(waypoint)
    waypoints.append(end)
    waypoints = np.array(waypoints)
    best_waypoints = waypoints.copy()
    best_cost = route_cost(best_waypoints, accident_zones, coeffs)
    for _ in range(iterations):
        new_waypoints = best_waypoints.copy()
        for i in range(1, len(new_waypoints)-1):
            perturbation = np.random.uniform(-step_size, step_size, size=2)
            new_waypoints[i] += perturbation
        new_cost = route_cost(new_waypoints, accident_zones, coeffs)
        if new_cost.item() < best_cost.item():
            best_cost = new_cost
            best_waypoints = new_waypoints.copy()
    return best_waypoints, best_cost

def plan_route(coeffs, num_accident_zones=1):
    accident_zones = create_accident_zones(num_zones=num_accident_zones, radius=0.04)
    best_waypoints, best_cost = optimize_route(denver, fort_collins, accident_zones, coeffs)
    return best_waypoints, best_cost, accident_zones

# --------------------------
# Reward Function
# --------------------------
def compute_reward(waypoints, accident_zones, coeffs):
    D0 = np.linalg.norm(fort_collins - denver)
    if waypoints is None:
        return -5.0
    total_distance = compute_total_distance(waypoints)
    total_trip_time = 60 + (total_distance - D0)
    inter_length = total_intersection_length(waypoints, accident_zones)
    intersection_penalty = 1000 * inter_length
    cost = coeffs[1] * total_distance + coeffs[2] * total_trip_time + coeffs[0] * intersection_penalty + 1e-6
    raw_reward = 1000 * (D0 / cost)
    reward = np.clip(raw_reward / 200.0, -5, 5)
    print(f"Reward: {reward:.3f}, Raw: {raw_reward:.3f}, Intersection Length: {inter_length:.3f}, Distance: {total_distance:.3f}, TripTime: {total_trip_time:.3f}")
    return reward

# --------------------------
# Visualization Function using OpenCV
# --------------------------
def plot_route(waypoints, accident_zones, cost, coeffs):
    fig, ax = plt.subplots(figsize=(8,6))
    if waypoints is not None:
        ax.plot(waypoints[:,1], waypoints[:,0], 'bo-', linewidth=3, label="Route")
        ax.scatter(waypoints[0,1], waypoints[0,0], color='green', s=100, label="Start")
        ax.scatter(waypoints[-1,1], waypoints[-1,0], color='red', s=100, label="End")
    for az in accident_zones:
        az_lat, az_lon, az_radius = az
        circle = plt.Circle((az_lon, az_lat), az_radius, color='black', alpha=0.5)
        ax.add_artist(circle)
        ax.text(az_lon, az_lat, "Accident", fontsize=10, ha='center', va='center', color='white')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Route (Cost: {cost:.2f})\nCoeffs: {np.round(coeffs,2)}")
    ax.legend()
    fig.canvas.draw()
    buf = fig.canvas.tostring_argb()
    ncols, nrows = fig.canvas.get_width_height()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 4)[:, :, 1:4]
    plt.close(fig)
    cv2.imshow("Route", img)
    cv2.waitKey(2000)
    cv2.destroyWindow("Route")

# --------------------------
# Custom Environment (Accident Avoidance Only)
# --------------------------
class RoadTripEnv(gym.Env):
    def __init__(self, render_enabled=False):
        super(RoadTripEnv, self).__init__()
        # Observation: rendered image of zones and route.
        self.observation_space = spaces.Box(low=0, high=255, shape=(84,84,3), dtype=np.uint8)
        # Action: 3 coefficients.
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.render_enabled = render_enabled
        self.accident_zones = create_accident_zones(num_zones=1, radius=0.01)
        self.last_route = None
        self.last_cost = None
        self.last_reward = None

    def get_image(self):
        fig, ax = plt.subplots(figsize=(4,4))
        # Draw fixed accident zone.
        for az in self.accident_zones:
            az_lat, az_lon, az_radius = az
            circle = plt.Circle((az_lon, az_lat), az_radius, color='black', alpha=0.5)
            ax.add_artist(circle)
        # If a route exists, plot it.
        if self.last_route is not None:
            ax.plot(self.last_route[:,1], self.last_route[:,0], 'bo-', linewidth=3, label="Route")
        ax.set_xlim(denver[1]-0.1, fort_collins[1]+0.1)
        ax.set_ylim(denver[0]-0.1, fort_collins[0]+0.1)
        ax.axis('off')
        fig.canvas.draw()
        buf = fig.canvas.tostring_argb()
        ncols, nrows = fig.canvas.get_width_height()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 4)[:, :, 1:4]
        plt.close(fig)
        img = cv2.resize(img, (84,84))
        return img

    def reset(self, seed=None, options=None):
        # Use fixed accident zone.
        self.accident_zones = create_accident_zones(num_zones=1, radius=0.01)
        self.last_route = None
        self.last_cost = None
        self.last_reward = None
        obs = self.get_image()
        return obs, {}

    def step(self, action):
        print("Agent's coefficients:", action)
        waypoints, cost, accident_zones = plan_route(action, num_accident_zones=1)
        self.last_route = waypoints
        self.last_cost = cost
        reward = compute_reward(waypoints, accident_zones, np.array(action).flatten())
        self.last_reward = reward
        obs = self.get_image()
        done = True
        truncated = False
        info = {"route": waypoints, "cost": cost, "reward": reward}
        if self.render_enabled:
            plot_route(waypoints, accident_zones, cost, np.array(action).flatten())
        return obs, reward, done, truncated, info

# --------------------------
# Register Environment with RLlib
# --------------------------
from ray.tune.registry import register_env
def env_creator(env_config):
    render_enabled = env_config.get("render_enabled", True)
    return RoadTripEnv(render_enabled=render_enabled)
register_env("RoadTripEnv-v0", env_creator)

# --------------------------
# (Optional) Register Custom Model with RLlib
# --------------------------
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model("custom_conv_model", CustomConvModel)

# --------------------------
# RLlib PPOConfig Setup (Legacy API)
# --------------------------
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

ray.shutdown()
ray.init(ignore_reinit_error=True)

config = (
    PPOConfig()
    .environment(env="RoadTripEnv-v0", env_config={"render_enabled": True})
    .training(
         use_critic=True,
         use_gae=True,
         lambda_=1.0,
         kl_coeff=0.2,
    )
    .resources(num_gpus=0)
    .env_runners(num_env_runners=1)
)
# For simplicity, we do not use a custom model in this version.
config.model = {
    "custom_preprocessor": None,
    "_disable_preprocessor_api": True,
    "max_seq_len": 20,
}
config = config.api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)

algo = config.build()

for i in range(10):
    result = algo.train()
    print(pretty_print(result))

policy = algo.get_policy()
test_env = RoadTripEnv(render_enabled=True)
obs, _ = test_env.reset()
action = policy.compute_single_action(obs)
print("Selected Coefficients:", action)
obs, reward, done, truncated, info = test_env.step(action)
print("Route Cost:", info["cost"])
print("Reward:", info["reward"])
