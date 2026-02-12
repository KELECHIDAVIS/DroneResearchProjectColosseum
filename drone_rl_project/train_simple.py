# train_simple.py
# ============================================================
# IMPORTS
# ============================================================

import setup_path         # Tells Python where to find the local AirSim library
                          # that matches the Blocks binary version. MUST come
                          # before importing airsim.

from stable_baselines3 import DQN
# DQN = Deep Q-Network. This is the RL algorithm we're using.
# It works by learning a "Q-value" for every (state, action) pair,
# which represents: "How much total future reward will I get if I
# take this action in this state?" The drone learns to always pick
# the action with the highest Q-value.
# Good choice here because: we have discrete actions (7 choices)
# and image-based observations (depth camera).

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
        # Called after EVERY single environment step during training.
        # self.locals contains variables from the training loop like:
        #   - rewards: the reward received this step
        #   - dones: whether the episode ended this step
        
        # Update progress bar by 1 step
        self.pbar.update(1)

        # Accumulate reward for this episode
        reward = self.locals.get("rewards", [0])[0]
        self.current_episode_reward += reward

        # Check if this episode is done
        done = self.locals.get("dones", [False])[0]
        if done:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)

            # Calculate average reward over last 10 episodes
            avg_reward = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards))

            # Update the progress bar description with live stats
            self.pbar.set_postfix({
                "episode": self.episode_count,
                "ep_reward": f"{self.current_episode_reward:.1f}",
                "avg_reward": f"{avg_reward:.1f}"
            })

            self.current_episode_reward = 0  # Reset for next episode

        return True  # Returning False would stop training early

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

total_timesteps = 10000
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

model = DQN(
    "CnnPolicy",
    # The neural network architecture to use.
    # "CnnPolicy" = Convolutional Neural Network policy.
    # CNN is ideal for image-based observations (like our depth camera)
    # because it can detect spatial patterns like edges, obstacles, etc.
    # Other options: "MlpPolicy" (for non-image observations like position vectors)

    env,
    # The environment the model will interact with and learn from.

    verbose=1,
    # Controls how much info SB3 prints during training.
    # 0 = silent, 1 = training info, 2 = debug info.

    learning_rate=1e-4,
    # How big each update step is when adjusting the neural network weights.
    # 1e-4 = 0.0001. Too high = unstable learning. Too low = slow learning.
    # Think of it like adjusting how much each mistake changes your behavior.

    buffer_size=10000,
    # Size of the "replay buffer" - a memory bank of past experiences.
    # DQN stores (state, action, reward, next_state) tuples here.
    # It randomly samples from this buffer to train, which breaks
    # correlations between consecutive experiences (important for stability).

    learning_starts=1000,
    # Number of steps to take BEFORE starting to train the neural network.
    # During these steps, the drone takes random actions to fill the buffer.
    # This ensures we have enough diverse experiences before learning starts.

    batch_size=32,
    # How many experiences to sample from the replay buffer per training update.
    # Larger = more stable but slower. 32 is a common default.

    tau=1.0,
    # Controls how the "target network" gets updated.
    # DQN uses TWO networks: the main network (being trained) and a 
    # target network (used as a stable reference for calculating Q-values).
    # tau=1.0 means hard copy (copy all weights at once).
    # tau<1.0 means soft update (blend weights gradually).

    gamma=0.99,
    # The "discount factor" - how much the agent cares about future rewards
    # vs immediate rewards. Range: 0 to 1.
    # gamma=0.99: future rewards matter almost as much as immediate ones.
    # gamma=0.0: only care about immediate rewards (short-sighted).
    # gamma=1.0: future rewards count exactly as much as current ones.

    train_freq=4,
    # How often to perform a training update.
    # train_freq=4 means: take 4 steps in the environment, then do 1 update.
    # Lower = more updates (slower but potentially better learning).

    target_update_interval=1000,
    # How many steps between copying the main network to the target network.
    # More frequent = more stable reference but slower to propagate learning.

    # ADD THESE for better exploration:
    exploration_fraction=0.3,
    # Spend 30% of training exploring randomly.
    # Default is 0.1 (10%) which might not be enough
    # to discover the "go up" strategy.

    exploration_initial_eps=1.0,
    # Start with 100% random actions (full exploration).

    exploration_final_eps=0.05,
    # End with 5% random actions (mostly using learned policy).
    # This decay happens over exploration_fraction * total_timesteps steps.

    tensorboard_log="./logs/"
    # Where to save training metrics for visualization with TensorBoard.
    # Run: tensorboard --logdir ./logs/
    # Then open http://localhost:6006 in your browser to see live charts.
)


# ============================================================
# TRAINING
# ============================================================

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

    log_interval=100,
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