# simple_obstacle_env.py
import setup_path
import airsim
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

class SimpleObstacleAvoidanceEnv(gym.Env):
    """
    Drone obstacle avoidance environment using AirSim/Colosseum.

    Design philosophy:
    - Reward is shaped to produce ONE clear gradient: move forward along X.
    - Altitude is softly constrained (not rewarded), so the drone has no
      incentive to climb to avoid obstacles — it must navigate around them.
    - The survive bonus is removed entirely. Every step that doesn't make
      forward progress costs slightly (via step penalty), forcing the agent
      to move rather than hover.
    - Generalizes to any map: goal_x, cruise_altitude, and bounds are all
      constructor parameters you can change per-map.
    """

    def __init__(
        self,
        goal_x: float = 50.0,          # Target X position in meters
        cruise_altitude: float = -3.0,  # NED Z (negative = above ground)
        altitude_tolerance: float = 2.0,# How far from cruise alt before penalty
        x_min: float = -5.0,           # Kill zone: too far backward
        max_steps: int = 200,          # Max steps per episode (safety kill switch) 
    ):
        super().__init__()

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # -----------------------------------------------------------------
        # Action space  (discrete is fast; 7 actions covers all directions)
        # -----------------------------------------------------------------
        # 0: forward   (+X)
        # 1: backward  (-X)
        # 2: left       (+Y in NED, drone-frame left)
        # 3: right      (-Y)
        # 4: up         (-Z, gain altitude)
        # 5: down       (+Z, lose altitude)
        # 6: hover      (no velocity)
        self.action_space = spaces.Discrete(7)

        # -----------------------------------------------------------------
        # Observation space  (84×84 depth image, 1 channel)
        # -----------------------------------------------------------------
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(84, 84, 1),
            dtype=np.uint8
        )

        # Episode parameters
        self.goal_x           = goal_x
        self.cruise_altitude  = cruise_altitude          # NED Z at reset
        self.altitude_tolerance = altitude_tolerance
        self.x_min            = x_min
        self.max_steps        = max_steps

        # Will be set in reset()
        self.current_step = 0
        self.last_x       = 0.0

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.client.takeoffAsync().join()

        # Fly to a consistent start pose
        self.client.moveToPositionAsync(
            x=0, y=0, z=self.cruise_altitude, velocity=3
        ).join()
        self.client.hoverAsync().join()

        self.current_step = 0
        self.last_x       = 0.0

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------------
    def _get_obs(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest(
                "0", airsim.ImageType.DepthPerspective,
                pixels_as_float=True, compress=False
            )
        ])
        response = responses[0]
        img1d = np.array(response.image_data_float, dtype=np.float32)
        img2d = img1d.reshape(response.height, response.width)

        # Clip to 100 m range and map to [0, 255]
        img2d = np.clip(img2d * 255.0 / 100.0, 0, 255).astype(np.uint8)
        img_resized = cv2.resize(img2d, (84, 84))

        return img_resized.reshape(84, 84, 1)

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------
    def step(self, action):
        self.current_step += 1

        # Velocity commands (vx, vy, vz) in world NED frame
        velocity_map = {
            0: ( 2,  0,  0),   # forward
            1: (-2,  0,  0),   # backward
            2: ( 0,  2,  0),   # left
            3: ( 0, -2,  0),   # right
            4: ( 0,  0, -2),   # up   (NED: -Z is up)
            5: ( 0,  0,  2),   # down (NED: +Z is down)
            6: ( 0,  0,  0),   # hover
        }

        vx, vy, vz = velocity_map[action]
        self.client.moveByVelocityAsync(vx, vy, vz, duration=0.2).join()
        

        # ---- Get state -----------------------------------------------
        state     = self.client.getMultirotorState()
        pos       = state.kinematics_estimated.position
        current_x = pos.x_val
        current_z = pos.z_val   # NED altitude (negative = above ground)
        collision = self.client.simGetCollisionInfo()

        # ---- Reward shaping ------------------------------------------
        done   = False
        reward = 0.0

        if collision.has_collided:
            # ── Terminal: collision ──────────────────────────────────
            reward = -100.0
            done   = True

        elif current_x < self.x_min:
            # ── Terminal: drone flew too far backward (lost) ─────────
            reward = -50.0
            done   = True

        else:
            # ── 1. Forward progress reward ───────────────────────────
            # This is the ONLY positive signal (besides goal bonus).
            # Scale: each 1 m of forward progress = +10 reward.
            x_progress = current_x - self.last_x
            reward += x_progress * 10.0

            # ── 2. Step penalty (replaces survive bonus) ─────────────
            # Small constant cost per step. This means hovering in place
            # costs −0.2/step while moving forward earns ~+10/step, so
            # the optimal policy is always to move forward.
            # It also prevents the drone from just climbing forever,
            # because climbing costs steps without earning progress.
            reward -= 0.2

            # ── 3. Altitude penalty (soft constraint) ─────────────────
            # If the drone drifts too far from cruise altitude, nudge it
            # back. The penalty is proportional to deviation so it
            # doesn't completely dominate the reward signal.
            alt_error = abs(current_z - self.cruise_altitude)
            if alt_error > self.altitude_tolerance:
                reward -= (alt_error - self.altitude_tolerance) * 1.0

            # ── 4. Goal bonus ─────────────────────────────────────────
            if current_x >= self.goal_x:
                reward += 500.0
                done    = True

        self.last_x = current_x

        if self.current_step >= self.max_steps:
            done = True

        obs = self._get_obs()
        return obs, reward, done, False, {
            "position_x": current_x,
            "position_z": current_z,
        }