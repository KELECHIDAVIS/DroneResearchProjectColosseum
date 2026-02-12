# train_simple.py
# ============================================================
# IMPORTS
# ============================================================

import setup_path         # Tells Python where to find the local AirSim library
                          # that matches the Blocks binary version. MUST come
                          # before importing airsim.

from stable_baselines3 import PPO 

from stable_baselines3.common.vec_env import DummyVecEnv
# SB3 (Stable Baselines 3) requires environments to be "vectorized"
# meaning it can run multiple environments in parallel.
# DummyVecEnv wraps our single environment to satisfy this requirement.
# Even though we only have 1 environment, we still need to wrap it.

from stable_baselines3.common.callbacks import BaseCallback
# BaseCallback lets us create custom functions that run during training.
# We'll use this to show a progress bar and log training stats.

from simple_obstacle_env import SimpleObstacleAvoidanceEnv
# Our custom Gymnasium environment that wraps AirSim.
# This is where the drone simulation lives.

import time
# Standard Python library for adding delays (time.sleep)

# ============================================================
# PROGRESS BAR CALLBACK
# ============================================================

from tqdm import tqdm
# tqdm is the progress bar library. Install with: pip install tqdm
# It wraps any iterable or can be manually updated to show progress.

class TqdmCallback(BaseCallback):
    # BaseCallback is SB3's way of letting you hook into the training loop.
    # Methods like _on_training_start and _on_step are called automatically
    # by SB3 at the right moments during training.

    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.episode_rewards = []      # Track rewards per episode
        self.current_episode_reward = 0
        self.episode_count = 0

    def _on_training_start(self):
        # Called ONCE when training begins.
        # We create the progress bar here with the total number of steps.
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training",           # Label shown on the left
            unit="steps",              # Unit label shown on the right
            colour="green",            # Bar color
            dynamic_ncols=True         # Adjusts to terminal width
        )

    def _on_step(self) -> bool:
        self.pbar.update(1)
        
        # PPO stores rewards differently than DQN
        # Use infos instead of rewards
        infos = self.locals.get("infos", [{}])
        
        if len(infos) > 0:
            # Check for episode end info
            if "episode" in infos[0]:
                ep_reward = infos[0]["episode"]["r"]
                self.episode_count += 1
                self.episode_rewards.append(ep_reward)
                
                avg_reward = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards))
                
                self.pbar.set_postfix({
                    "episode": self.episode_count,
                    "ep_reward": f"{ep_reward:.1f}",
                    "avg_reward": f"{avg_reward:.1f}"
                })
        
        return True
    
    def _on_training_end(self):
        # Called ONCE when training finishes.
        # Close the progress bar cleanly.
        self.pbar.close()
        print(f"\nTraining complete! Total episodes: {self.episode_count}")
        if self.episode_rewards:
            print(f"Best episode reward: {max(self.episode_rewards):.1f}")
            print(f"Final avg reward (last 10): {sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards)):.1f}")


# ============================================================
# ENVIRONMENT SETUP
# ============================================================

# total_timesteps = 10000
# Total number of steps the drone will take during ALL of training.
# Each "step" = one action taken in the environment.
# 10,000 is small (quick demo). Real training might use 1,000,000+.

print("Setting up environment...")

env = SimpleObstacleAvoidanceEnv()
# Creates an instance of our custom AirSim Gym environment.
# This connects to the Blocks simulation.

env = DummyVecEnv([lambda: env])
# Wraps our environment in SB3's vectorized environment format.
# The lambda: env syntax creates a function that returns our env.
# DummyVecEnv expects a LIST of environment-creating functions,
# which is why we pass [lambda: env] (a list with one function).


# ============================================================
# MODEL SETUP
# ============================================================
import torch_directml
device = torch_directml.device()
print(f"Training on: {device}")

model = PPO(
    "CnnPolicy",  # Use a convolutional neural network (CNN) policy,
    env,
    device=device,  # Use GPU if available for faster training
    verbose=0,    # 0 = no output, 1 = info, 2 = debug
    learning_rate=3e-4,  # How quickly the model updates its knowledge each step
    n_steps=512,
    batch_size=64,
    n_epochs=10, 
    gamma=0.99,    # Discount factor for future rewards
    gae_lambda=0.95, # GAE lambda for advantage estimation
    clip_range=0.2,  # prevents too large updates to the policy (stabilizes training) 
    ent_coef= 0.01, # Encourages exploration by adding a small bonus to the loss for having higher entropy (more randomness in action selection) 
    tensorboard_log="./logs/"
)


# ============================================================
# TRAINING
# ============================================================
total_timesteps = 100000 
print("Starting training...")
print("Watch the Blocks window to see the drone learning!")
print(f"Training for {total_timesteps} steps...\n")

# Create our progress bar callback
progress_callback = TqdmCallback(total_timesteps)

model.learn(
    total_timesteps=total_timesteps,
    # Total number of steps to train for.

    callback=progress_callback,
    # Our custom callback that shows the progress bar and stats.

    log_interval=1,
    # How often (in episodes) to log training info.

    reset_num_timesteps=True,
    # Whether to reset the step counter when calling learn().
    # True = fresh start. False = continue from where we left off.
    # Useful when continuing training: model.learn(..., reset_num_timesteps=False)
)

model.save("obstacle_avoidance_simple")
# Saves the trained model weights to a file called "obstacle_avoidance_simple.zip"
# Load it later with: model = DQN.load("obstacle_avoidance_simple")


# ============================================================
# QUICK TEST OF TRAINED MODEL
# ============================================================

print("\nTraining complete! Running quick test...")
print("Watch the Blocks window to see the trained drone!\n")

obs = env.reset()
# Reset the environment to start a fresh episode.
# Returns the first observation (depth image from camera).

total_reward = 0

for i in range(200):
    # Run up to 200 steps to test the trained agent.

    action, _states = model.predict(obs, deterministic=True)
    # Ask the trained model: "Given this observation, what action should I take?"
    # deterministic=True: always pick the BEST action (no randomness).
    #                     During training, some randomness is used to explore.
    #                     During testing, we want the best learned behavior.
    # _states: only used for recurrent policies (like LSTM), not needed here.

    obs, reward, done, info = env.step(action)
    # Execute the action in the environment.
    # Returns:
    #   obs    = new observation (next depth image)
    #   reward = reward received for this step
    #   done   = True if episode ended (collision or goal reached)
    #   info   = extra info dict (we put position_x here)

    total_reward += reward[0]  # reward is a list in VecEnv, take first element

    time.sleep(0.1)
    # Small delay so we can watch the drone move in Blocks.
    # Remove this during actual training to speed things up.

    if done[0]:
        # done is also a list in VecEnv, check first element.
        x_pos = info[0].get('position_x', 0)
        print(f"Episode finished at step {i}!")
        print(f"Final position X: {x_pos:.2f} meters")
        print(f"Total reward: {total_reward:.1f}")

        if x_pos >= 50:
            print("SUCCESS! Drone reached the goal!")
        else:
            print("Drone didn't reach goal (collision or timeout)")
        break

env.close()
# Cleanly close the environment and disconnect from AirSim.
print("\nTest complete!")