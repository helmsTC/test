import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import uuid, queue, threading

class CustomConvModel(nn.Module):
    """Tiny 3‑layer CNN that squeezes an 84×84 RGB image to an 8‑D feature vector.
    Plug it into RLlib/Stable‑Baselines if you need a custom encoder.
    """

    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 8,  kernel_size=9, stride=1)

    def forward(self, obs: torch.Tensor):  # obs: [B,3,84,84]
        x = obs.float() / 255.0
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return x.view(x.size(0), -1)  # [B,8]


class ImageActionEnv(gym.Env):
    """Stateless env that *asynchronously* receives rendered images.

    * Action → 4‑D Box(0,1).
    * Reward comes back later with the rendered frame.
    * No blocking inside ``step()`` – it merely *queues* the request.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_enabled: bool = False):
        super().__init__()
        self.observation_space = spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8)
        self.action_space      = spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)
        self.render_enabled    = render_enabled

        # Async plumbing --------------------------------------------------
        self._request_q: queue.Queue[tuple[str, np.ndarray]] = queue.Queue()
        self._last_obs     = np.zeros((84, 84, 3), dtype=np.uint8)
        self._last_reward  = 0.0

    # ---------------------------------------------------------------
    # Public handles for the outside world
    # ---------------------------------------------------------------
    @property
    def request_queue(self):
        """Queue of (req_id, action) tuples waiting to be rendered."""
        return self._request_q

    def deliver(self, req_id: str, img: np.ndarray, reward: float):
        """Called by *external* code when the renderer finishes."""
        img = cv2.resize(img, (84, 84)) if img.shape[:2] != (84, 84) else img
        self._last_obs    = img.astype(np.uint8)
        self._last_reward = float(reward)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._last_obs    = np.zeros((84, 84, 3), dtype=np.uint8)
        self._last_reward = 0.0
        return self._last_obs.copy(), {}

    def step(self, action: np.ndarray):
        # 1) push request for rendering (non‑blocking)
        req_id = uuid.uuid4().hex
        self._request_q.put((req_id, np.clip(action, 0.0, 1.0).copy()))

        # 2) emit *previous* obs/reward; they will be replaced asynchronously
        obs, reward = self._last_obs.copy(), self._last_reward
        self._last_reward = 0.0  # consume it once returned
        info = {"request_id": req_id}
        return obs, reward, False, False, info

    def render(self):
        if self.render_enabled:
            cv2.imshow("ImageActionEnv", cv2.cvtColor(self._last_obs, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        return self._last_obs.copy()

    def close(self):
        if self.render_enabled:
            cv2.destroyAllWindows()

class RandomPolicy:
    """Stub policy that samples uniformly from the action space."""
    def __init__(self, act_space):
        self.act_space = act_space
    def act(self, obs):
        return self.act_space.sample()

class BackendExample:
    """Shows the full async loop without blocking the main thread."""

    def __init__(self):
        self.env    = ImageActionEnv(render_enabled=False)
        self.policy = RandomPolicy(self.env.action_space)

        # Spin up a background thread that *services* render jobs.
        self.worker = threading.Thread(target=self._render_worker, daemon=True)
        self.worker.start()

        # Initial observation
        self.obs, _ = self.env.reset()

    def _render_worker(self):
        """Pretend render process – replace with Redis, gRPC, etc."""
        while True:
            req_id, action = self.env.request_queue.get()  # blocks until next job
            # ---- heavy path‑drawing code would run here ----------------
            img = np.random.randint(0, 256, (84, 84, 3), np.uint8)  # fake path image
            reward = 1.0                                           # fake reward
            # Deliver back to the env (thread‑safe)
            self.env.deliver(req_id, img, reward)

    # -----------------------------------------------------------------
    def train_loop(self, steps: int = 10_000):
        for _ in range(steps):
            action = self.policy.act(self.obs)
            self.obs, reward, _, _, info = self.env.step(action)
            # `reward` corresponds to *previous* image; use it to train
            # (Here we just print to prove it flows.)
            if reward != 0.0:
                print(f"Got reward {reward:.1f} for request {info.get('request_id')}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    backend = BackendExample()
    backend.train_loop(steps=1000)
