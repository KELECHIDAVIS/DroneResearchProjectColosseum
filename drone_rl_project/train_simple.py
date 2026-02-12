# train_simple.py
# ============================================================
# IMPORTS
# ============================================================

import setup_path          # MUST come before `import airsim`

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
# CallbackList    – run multiple callbacks together
# CheckpointCallback – auto-save the model every N steps
# EvalCallback    – periodically run the greedy policy and log mean reward;
#                   this is what actually writes episode stats to TensorBoard

from stable_baselines3.common.monitor import Monitor
# Monitor wraps the env and records episode reward/length in `info["episode"]`.
# THIS IS THE KEY FIX FOR TENSORBOARD: SB3's built-in logger only writes
# episode-level stats when it sees info["episode"], which Monitor provides.
# Without Monitor, the tensorboard_log= folder is created but stays empty.

from simple_obstacle_env import SimpleObstacleAvoidanceEnv

import time
from tqdm import tqdm


# ============================================================
# PROGRESS BAR CALLBACK
# ============================================================

class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps    = total_timesteps
        self.pbar               = None
        self.episode_rewards    = []
        self.episode_count      = 0

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training",
            unit="steps",
            colour="green",
            dynamic_ncols=True,
        )

    def _on_step(self) -> bool:
        self.pbar.update(1)

        # Monitor puts episode stats in info["episode"] at episode end
        infos = self.locals.get("infos", [{}])
        if infos and "episode" in infos[0]:
            ep_reward = infos[0]["episode"]["r"]
            self.episode_count += 1
            self.episode_rewards.append(ep_reward)
            avg = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards))
            self.pbar.set_postfix(
                episode=self.episode_count,
                ep_reward=f"{ep_reward:.1f}",
                avg10=f"{avg:.1f}",
            )
        return True

    def _on_training_end(self):
        self.pbar.close()
        print(f"\nTraining complete!  Episodes: {self.episode_count}")
        if self.episode_rewards:
            print(f"  Best reward : {max(self.episode_rewards):.1f}")
            print(f"  Last-10 avg : {sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards)):.1f}")


# ============================================================
# ENVIRONMENT FACTORY
# ============================================================
# We use a factory function (make_env) rather than a raw lambda so
# that each environment is independently wrapped with Monitor.
# This pattern also makes it easy to swap to SubprocVecEnv later
# if you want to run multiple AirSim instances in parallel.

def make_env(goal_x=50.0, log_dir="./monitor_logs/"):
    def _init():
        env = SimpleObstacleAvoidanceEnv(goal_x=goal_x)
        # Monitor logs episode reward + length and injects info["episode"].
        # log_dir=None means it only tracks in-memory (no extra CSV file
        # written alongside your logs folder). Set a path if you want CSVs.
        env = Monitor(env, filename=None)
        return env
    return _init


# ============================================================
# MAIN SETUP
# ============================================================

TOTAL_TIMESTEPS = 70000   # Start here; bump to 1M+ for serious training
GOAL_X          = 50.0
LOG_DIR         = "./logs/"
CHECKPOINT_DIR  = "./checkpoints/"

print("Setting up environment…")

# DummyVecEnv runs one environment synchronously.
# The factory pattern is correct here: pass a list of callables, not instances.
env = DummyVecEnv([make_env(goal_x=GOAL_X)])


# ============================================================
# MODEL
# ============================================================

model = PPO(
    "CnnPolicy",
    env,
    verbose=0,           # 0 = silent; set to 1 to see SB3's own log lines
    learning_rate=3e-4,
    n_steps=1024,        # Collect 1024 steps before each update.
                         # With max_steps=500, this captures ~2 full episodes
                         # per update, which stabilises the advantage estimates.
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,       # Small entropy bonus keeps exploration alive.
                         # If the drone stops exploring, raise to 0.02-0.05.
    tensorboard_log=LOG_DIR,
)

# ============================================================
# CALLBACKS
# ============================================================

# 1. Auto-save every 25 k steps so you never lose a good run
checkpoint_cb = CheckpointCallback(
    save_freq=25_000,
    save_path=CHECKPOINT_DIR,
    name_prefix="obstacle_ppo",
    verbose=1,
)

# 2. Progress bar in the terminal
progress_cb = TqdmCallback(TOTAL_TIMESTEPS)

# Combine all callbacks. Add EvalCallback here if you want a separate
# eval env (requires a second AirSim connection or pausing the train env).
callbacks = CallbackList([checkpoint_cb, progress_cb])


# ============================================================
# TRAINING
# ============================================================

print("Starting training…")
print(f"Goal: reach X = {GOAL_X} m")
print(f"Steps: {TOTAL_TIMESTEPS:,}")
print(f"TensorBoard: run `tensorboard --logdir {LOG_DIR}` in another terminal\n")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callbacks,
    log_interval=1,        # Log every episode (Monitor makes this meaningful)
    reset_num_timesteps=True,
    tb_log_name="blocks_ppo",   # Sub-folder name inside LOG_DIR
)

model.save("obstacle_avoidance_ppo")
print("\nModel saved to obstacle_avoidance_ppo.zip")


# ============================================================
# QUICK EVALUATION RUN
# ============================================================

print("\nRunning evaluation episode…\n")

obs = env.reset()
total_reward = 0.0

for i in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward[0]
    time.sleep(0.05)   # Slow down just enough to watch; remove to go faster

    if done[0]:
        x_pos = info[0].get("position_x", 0)
        print(f"Episode ended at step {i+1}")
        print(f"  Final X  : {x_pos:.2f} m")
        print(f"  Reward   : {total_reward:.1f}")
        print(f"  Outcome  : {'GOAL REACHED' if x_pos >= GOAL_X else 'Did not reach goal'}")
        break

env.close()
print("\nDone.")