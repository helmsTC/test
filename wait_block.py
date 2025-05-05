import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Off‑screen rendering
import cv2, torch, torch.nn as nn, torch.nn.functional as F
import uuid, queue, threading, concurrent.futures, time

# =====================================================================================
#                       Optional CNN image encoder
# =====================================================================================

class CustomConvModel(nn.Module):
    """Simple 3‑layer CNN that compresses an 84×84 RGB image to 8 features."""

    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.conv3 = nn.Conv2d(32, 8, 9, 1)

    def forward(self, obs):
        x = obs.float() / 255.0
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return x.view(x.size(0), -1)

# =====================================================================================
#                 Non‑blocking image/4‑action Gym environment
# =====================================================================================

class ImageActionEnv(gym.Env):
    """Env returns previous obs/reward while a new render is prepared externally."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_enabled: bool = False):
        super().__init__()
        self.observation_space = spaces.Box(0, 255, (84, 84, 3), np.uint8)
        self.action_space      = spaces.Box(0.0, 1.0, (4,), np.float32)
        self.render_enabled    = render_enabled

        self._last_obs    = np.zeros((84, 84, 3), np.uint8)
        self._last_reward = 0.0

    # Gym boilerplate --------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._last_obs[:] = 0
        self._last_reward = 0.0
        return self._last_obs.copy(), {}

    def step(self, action):
        # No internal queues any more – external Backend decides when to call deliver()
        obs, rew = self._last_obs.copy(), self._last_reward
        self._last_reward = 0.0
        info = {}
        return obs, rew, False, False, info

    # External process feeds results here -----------------------------
    def deliver(self, img: np.ndarray, reward: float):
        img = cv2.resize(img, (84, 84)) if img.shape[:2] != (84, 84) else img
        self._last_obs    = img.astype(np.uint8)
        self._last_reward = float(reward)

    def render(self):
        if self.render_enabled:
            cv2.imshow("ImageActionEnv", cv2.cvtColor(self._last_obs, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        return self._last_obs.copy()

    def close(self):
        if self.render_enabled:
            cv2.destroyAllWindows()

# =====================================================================================
#                         Plan‑designer component
# =====================================================================================

class PlanDesigner:
    """Heavy, blocking path‑generation logic (replace with real draw calls)."""

    def design(self, prefs: np.ndarray):
        # FAKE workload – imagine querying another service, drawing SVG, etc.
        time.sleep(0.05)  # simulate latency
        img = np.random.randint(0, 256, (84, 84, 3), np.uint8)
        reward = 1.0
        return img, reward

# =====================================================================================
#                             Policy stub
# =====================================================================================

class RandomPolicy:
    def __init__(self, act_space):
        self.act_space = act_space
    def act(self, obs):
        return self.act_space.sample()

# =====================================================================================
#                        Asynchronous backend orchestrator
# =====================================================================================

class Backend:
    """Represents `backend.py`.

    Flow per tick:
      1. **get_preferences()** – query policy → 4‑vector.
      2. **get_plans()** – submit heavy `plan_designer.design()` into a thread pool.
      3. When finished, results are pushed to env via `analysis_fit()`.
    """

    def __init__(self, max_workers: int = 4):
        self.env           = ImageActionEnv(render_enabled=False)
        self.policy        = RandomPolicy(self.env.action_space)
        self.plan_designer = PlanDesigner()

        self.pool    = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.pending: set[concurrent.futures.Future] = set()

        self.obs, _ = self.env.reset()

    # -----------------------------------------------------------------
    def get_preferences(self) -> np.ndarray:
        """Ask the agent for its 4‑D preference vector (non‑blocking)."""
        prefs = self.policy.act(self.obs)
        return prefs

    # -----------------------------------------------------------------
    def get_plans(self, prefs: np.ndarray):
        """Kick off path generation in the background – returns immediately."""
        future = self.pool.submit(self.plan_designer.design, prefs.copy())
        self.pending.add(future)

    # -----------------------------------------------------------------
    def _integrate_finished_jobs(self):
        done = {f for f in list(self.pending) if f.done()}
        for f in done:
            self.pending.remove(f)
            img, reward = f.result()
            self.env.deliver(img, reward)
            self.analysis_fit(img, reward)

    # -----------------------------------------------------------------
    def analysis_fit(self, img: np.ndarray, reward: float):
        """Here: update RL algorithm; now we just print."""
        print(f"analysis_fit: reward {reward:.1f}, img received")
        # Feed new obs to policy for next step
        self.obs = img

    # -----------------------------------------------------------------
    def run(self, iterations: int = 1000):
        for _ in range(iterations):
            prefs = self.get_preferences()   # 1) agent proposes
            self.get_plans(prefs)            # 2) submit heavy work
            self._integrate_finished_jobs()  # 3) handle finished renders
            time.sleep(0.01)                # main‑loop tick

# ---------------------------------------------------------------------
if __name__ == "__main__":
    backend = Backend(max_workers=4)
    backend.run(500)
