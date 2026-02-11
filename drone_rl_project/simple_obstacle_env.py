# simple_obstacle_env.py
import setup_path
import airsim
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

class SimpleObstacleAvoidanceEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        # Simple actions: [forward/back, left/right, up/down]
        # Discrete is MUCH faster to train than continuous
        self.action_space = spaces.Discrete(7)
        # 0: forward, 1: backward, 2: left, 3: right, 4: up, 5: down, 6: hover
        
        # Observation: just depth image (simplified)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(84, 84, 1),
            dtype=np.uint8
        )
        
        self.goal_x = 50  # Simple goal: reach x=50
        self.max_steps = 500
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Takeoff first
        self.client.takeoffAsync().join()
        
        # Then fly to consistent starting altitude and position
        self.client.moveToPositionAsync(
            x=0,       # Start at origin X
            y=0,       # Start at origin Y
            z=-3,      # 3 meters above ground (negative = up in NED)
            velocity=2 # Move at 2 m/s
        ).join()
        
        self.client.hoverAsync().join()  # Stabilize before episode starts
        
        self.current_step = 0
        self.last_x = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        # Get depth image
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, 
                              pixels_as_float=True, compress=False)
        ])
        
        response = responses[0]
        img1d = np.array(response.image_data_float, dtype=np.float32)
        img2d = img1d.reshape(response.height, response.width)
        
        # Normalize and resize to 84x84
        img2d = np.clip(img2d * 255 / 100, 0, 255).astype(np.uint8)
        img_resized = cv2.resize(img2d, (84, 84))
        
        return img_resized.reshape(84, 84, 1)
    
    def step(self, action):
        self.current_step += 1
        
        # Map discrete action to velocity
        velocity_map = {
            0: (2, 0, 0),    # forward
            1: (-2, 0, 0),   # backward
            2: (0, 2, 0),    # left
            3: (0, -2, 0),   # right
            4: (0, 0, -1),   # up
            5: (0, 0, 1),    # down
            6: (0, 0, 0)     # hover
        }
        
        vx, vy, vz = velocity_map[action]
        
        # Execute action
        self.client.moveByVelocityAsync(vx, vy, vz, duration=0.5).join()
        self.client.hoverAsync().join()
        
        # Get current position
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        current_x = pos.x_val
        
        # Check collision
        collision = self.client.simGetCollisionInfo()
        
        # Calculate reward
        reward = 0
        done = False
        
        if collision.has_collided:
            reward = -100
            done = True
        else:
            # Reward for moving forward (toward goal)
            progress = current_x - self.last_x
            reward = progress * 10  # Scale up the reward
            
            # Bonus for reaching goal
            if current_x >= self.goal_x:
                reward += 500
                done = True
        
        self.last_x = current_x
        
        # Timeout
        if self.current_step >= self.max_steps:
            done = True
        
        obs = self._get_obs()
        
        return obs, reward, done, False, {"position_x": current_x}