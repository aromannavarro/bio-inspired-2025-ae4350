import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from train_DQN import DQN

# === SETTINGS ===
MODEL_PATH = "saved_models/gamma_0.99_episode_514.pth"  # Adjust if needed
LAYERS = [128, 128]  # Must match the trained model architecture
EPISODES = 5
RENDER_MODE = "human"  # Use "rgb_array" for headless mode (e.g., saving frames)

environment_parameters = {
    "gravity": -10.0,
    "enable_wind": False,
    "wind_power": 5.0,
    "turbulence_power": 0.25
}
# === ENVIRONMENT ===
# Use the imported environment_parameters for consistency
env = gym.make("LunarLander-v3",
               continuous=False,
               gravity=environment_parameters["gravity"],
               enable_wind=environment_parameters["enable_wind"],
               wind_power=environment_parameters["wind_power"],
               turbulence_power=environment_parameters["turbulence_power"],
               render_mode=RENDER_MODE)

obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL ===
model = DQN(obs_dim, n_actions, LAYERS).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() # Set the model to evaluation mode

# === PLAY EPISODES ===
for ep in range(EPISODES):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        with torch.no_grad(): # Disable gradient calculations for inference
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            action = model(state).argmax().item() # Get the action with the highest Q-value
        
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated # Episode ends if terminated or truncated
        total_reward += reward
    print(f"Episode {ep + 1}: Reward = {total_reward:.2f}")

env.close() # Close the environment after running all episodes
