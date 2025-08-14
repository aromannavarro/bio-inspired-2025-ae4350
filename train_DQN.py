import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import random
import numpy as np
import os
import csv

from torch.utils.tensorboard import SummaryWriter
from collections import deque, namedtuple
from typing import Tuple, List, Dict, Any
from typing import Optional, Dict, Any

# --- DQN ---
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, layer_dim=[128, 128]):
        super(DQN, self).__init__()
        self.layers = nn.ModuleList()
        prev_dim = n_observations
        for dim in layer_dim:
            self.layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        self.output_layer = nn.Linear(prev_dim, n_actions)


    def forward(self, x):
        for layer in self.layers: # Iterate through the ModuleList directly to be able to work with different number of layers for tuning
            x = F.relu(layer(x))
        x = self.output_layer(x) # Apply output layer outside the loop if it's the only one without ReLU
        return x

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# --- ReplayBuffer ---
class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        if len(self.buffer) < batch_size:
            # This check is important! and will be the implicit 'min_replay_size'
            raise ValueError(f"Replay buffer has {len(self.buffer)} samples, but {batch_size} requested for sampling.")
        
        samples: List[Transition] = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device)
        rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def __len__(self) -> int:
        return len(self.buffer)

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_size, action_size, hidden_layers, learning_rate, gamma, batch_size, target_update_freq):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #check to make the process faster if possible

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.action_size = action_size

        self.policy_net = DQN(state_size, action_size, hidden_layers).to(self.device)
        self.target_net = DQN(state_size, action_size, hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        self.replay_buffer = ReplayBuffer(100000, self.device)
        self.frame_idx = 0 

    def act(self, obs, epsilon):
        if random.random() < epsilon:
            return random.choice(np.arange(self.action_size))
        else:
            with torch.no_grad():
                state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                action = self.policy_net(state).argmax().item()
                return action

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            # If not enough samples, no learning happens this step
            return None 

        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = self.replay_buffer.sample(self.batch_size)

        q_values = self.policy_net(obs_batch).gather(1, act_batch.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q_values = self.target_net(next_obs_batch).max(1)[0]
            target_q = rew_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.frame_idx += 1
        if self.frame_idx % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# --- Convergence Evaluation Function ---
def evaluate_convergence(
        csv_filepath: str,
        target_average_reward: float = 200,
        recent_window_size: int = 100,
        required_consecutive_successes: int = 90, 
        smoothing_alpha: float = 0.95, # For Exponential Moving Average (EMA)
        display_status: Optional[str] = None
    ) -> bool:
        
        episode_rewards = []
        try:
            with open(csv_filepath, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 2: # Ensure at least two columns (episode, reward)
                        try:
                            episode_rewards.append(float(row[1]))
                        except ValueError:
                            print(f"Warning: Malformed reward value '{row[1]}' in {csv_filepath}. Skipping row.")
                            continue
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_filepath}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred while reading {csv_filepath}: {e}")
            return False

        if len(episode_rewards) < recent_window_size:
            if display_status:
                print(f"{display_status} -- Not enough episodes ({len(episode_rewards)}) for "
                    f"evaluation window ({recent_window_size}).")
            return False

        recent_rewards = np.array(episode_rewards[max(0, len(episode_rewards) - recent_window_size):])

        #calculate metrics
        current_average_reward = np.mean(recent_rewards)

        # Define "success" for consistency check. Often, it's a high percentage of the target.
        #For LunarLander, 200 is "solved", so 180 or 190 is taken to be a good "consistent success" threshold
        consistency_reward_threshold = target_average_reward * 0.9 # e.g., 90% of target
        num_consistent_successes = np.sum(recent_rewards >= consistency_reward_threshold)

        if display_status:
            print(
                f"{display_status}, Avg reward (last {recent_window_size}): {current_average_reward:.2f}, "
                f"Consistent successes (>={consistency_reward_threshold:.1f}): {int(num_consistent_successes)}/{recent_window_size}"
            )

        # Convergence criterias:
        # Criterion 1: The average reward in the recent window meets the punctuation considered as succesful 
        avg_reward_met = current_average_reward >= target_average_reward

        # Criterion 2: A high number of episodes in the recent window consistently perform well, to avoid local minima
        consistency_met = num_consistent_successes >= required_consecutive_successes

        # Convergence is achieved when both conditions are met
        if avg_reward_met and consistency_met:
            return True
        
        return False

# --- Save information in the cv file for later ploting ---
def save_line(filename, episode, reward):
    with open(filename, 'a') as f:
        f.write(f"{episode},{reward}\n")

# --- Funcitno that will be called to train the agent, which calls all the previous funcitons ---
def trainDQN(
    env: gym.Env,
    agent: 'DQNAgent',
    writer: SummaryWriter,
    config: Dict[str, Any],
    csv_filename: str,
    iteration: str = None # For naming saved models/logs during tuning
):
    # Extract parameters from the config dictionary specified in tune_DQN.py
    max_episodes = config["max_episodes"]
    epsilon_start = config["epsilon_start"]
    epsilon_min = config["epsilon_min"]
    epsilon_decay = config["epsilon_decay"]
    #we still need batch_size from config for agent.learn()
    batch_size = config["batch_size"] 

    with open(csv_filename, 'w') as f:
        f.write("episode,reward\n") 

    episode_count = 0
    train_active = True
    obs, _ = env.reset() # Initial observation for the first step

    while train_active:
        current_epsilon = max(epsilon_min, epsilon_start - epsilon_decay * episode_count)

        episode_reward = 0.0
        done = False
        episode_losses = []

        # Episode loop
        while not done:
            # Action selection: initially pure random for exploration, then epsilon-greedy is used
            # For the very first frames, epsilon_start is 1.0, so it's purely random.
            action = agent.act(obs, current_epsilon) 
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs if not done else env.reset()[0] # Reset for next episode

            episode_reward += reward

            # Attempt to learn. The agent.learn() method itself will check buffer size.
            loss = agent.learn()
            if loss is not None:
                episode_losses.append(loss)
                writer.add_scalar("Loss/Frame", loss, agent.frame_idx)
            writer.add_scalar("Epsilon/Frame", current_epsilon, agent.frame_idx)
        
        # Episode ends here
        avg_episode_loss = np.mean(episode_losses) if episode_losses else 0.0
        
        # Log episode-level metrics
        save_line(csv_filename, episode_count, episode_reward)
        writer.add_scalar("Reward/Episode", episode_reward, episode_count)
        writer.add_scalar("Loss/Episode_Avg", avg_episode_loss, episode_count)
        writer.add_scalar("Epsilon/Episode", current_epsilon, episode_count)

        status_message = (
            f"Episode: {episode_count + 1}, "
            f"Reward: {int(episode_reward)}, "
        )
        # Check convergence
        if evaluate_convergence(csv_filename, display_status=status_message):
            train_active = False
            print(f"Convergence achieved at episode {episode_count + 1}.")
            converged_model_name = f"{iteration}_episode_{episode_count + 1}.pth" if iteration else f"converged_episode_{episode_count + 1}.pth"
            torch.save(agent.policy_net.state_dict(), os.path.join("saved_models", converged_model_name))
        
        episode_count += 1
        
        # In case convergence is not reached, a maximum number of episodes is given by the user.
        if episode_count >= max_episodes:
            train_active = False
            print(f"Maximum episodes ({max_episodes}) reached. Training stopped.")
            final_model_name = f"iteration_{iteration}_final_episode_{episode_count}.pth" if iteration else f"final_episode_{episode_count}.pth"
            torch.save(agent.policy_net.state_dict(), os.path.join("saved_models", final_model_name))

    print("Training complete.")