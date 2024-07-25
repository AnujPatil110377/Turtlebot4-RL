#!/usr/bin/env python3
"""
Training script for TurtleBot4 exploration using PPO algorithm.

This script implements the training pipeline for autonomous navigation
and exploration tasks using Proximal Policy Optimization (PPO).

Author: Anuj Patil
Based on original work by anurye (https://github.com/anurye/gym-turtlebot)
"""

import os
import sys
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Add the package path to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tb4_drl_navigation.envs.diffdrive.turtlebot4 import TurtleBot4Env


def main():
    """Main training function."""
    print("Starting TurtleBot4 PPO Training...")
    
    # Environment configuration
    env_config = {
        'reward_function': 'exploration',
        'max_episode_steps': 1000,
        'robot_name': 'turtlebot4',
        'world_name': 'static_world'
    }
    
    # Create environment
    env = TurtleBot4Env(**env_config)
    env = DummyVecEnv([lambda: env])
    
    # PPO model configuration
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./logs/ppo_turtlebot4/"
    )
    
    # Training callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/',
        name_prefix='ppo_turtlebot4'
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path='./models/best/',
        log_path='./logs/eval/',
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Start training
    print("Training PPO agent for exploration task...")
    model.learn(
        total_timesteps=100000,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save("./models/ppo_turtlebot4_final")
    print("Training completed! Model saved.")


if __name__ == "__main__":
    main()