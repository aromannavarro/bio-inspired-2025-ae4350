import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import random
import numpy as np
import gymnasium as gym
import os

from train_DQN import DQNAgent, DDQNAgent, trainDQN

# --- Configuration Flags ---
RUN_NOMINAL_BASELINE = False

# Tuning flags - set to True to run that specific tuning phase
# Take into consideration that if the code is ran with any of the conditions as True, the current existing csv file in tune/ will be overwritten
TUNE_LAYERS = False
TUNE_BATCH_SIZE = False
TUNE_LEARNING_RATE = False 
TUNE_EPSILON_DECAY = False
TUNE_EPSILON_MIN = False
TUNE_GAMMA = False
RUN_DDQN_TEST = True
RUN_ENVIRONMENT = False

# --- Directory Setup ---
os.makedirs("tune", exist_ok=True)
os.makedirs("tune/nominal", exist_ok=True)
os.makedirs("tune/learning_rate", exist_ok=True)
os.makedirs("tune/batch", exist_ok=True)
os.makedirs("tune/epsilon_decay", exist_ok=True)
os.makedirs("tune/epsilon_min", exist_ok=True) # Added for clarity
os.makedirs("tune/layers", exist_ok=True)
os.makedirs("tune/discount_factor", exist_ok=True)
os.makedirs("tune/environment", exist_ok=True)
os.makedirs("tune/ddqn_test", exist_ok=True) # New directory for DDQN results
os.makedirs("saved_models", exist_ok=True)

# fixe Seed for Reproducibility 
SEED = 42 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Nominal Hyperparameters ---
config = {
    "learning_rate": 1e-3,
    "batch_size": 64, # https://web.stanford.edu/class/aa228/reports/2019/final8.pdf
    "layers": [128, 128],
    "epsilon_start": 1.0,
    "epsilon_min": 0.01, # https://www.researchgate.net/profile/Xinli-Yu/publication/333145451_Deep_Q-Learning_on_Lunar_Lander_Game/links/5cdd8121299bf14d959dcc0d/Deep-Q-Learning-on-Lunar-Lander-Game.pdf
    "epsilon_decay": 0.002, 
    "gamma": 0.99, #stand 
    "target_update_freq": 1000,
    "max_episodes": 2000,
}

# --- Tuning Ranges for Each Parameter ---
TUNING_RANGES = {
    "layers": [ 
        [128, 64],
        [256, 128],
        [128, 128], # Nominal
        [512, 256],
        [256, 128, 64], 
        [512, 256, 128]
    ],
    "batch_size": [32, 64, 128, 256], 
    "learning_rate": [5e-5, 1e-4, 5e-4, 1e-3], 
    "epsilon_decay": [0.001, 0.002, 0.005, 0.01],
    "epsilon_min": [0.01, 0.02, 0.05], 
    "gamma": [0.98, 0.99, 0.995] 
}

# --- Environment Parameters ---
# This will be changed later to evaluate the performance of the agent with different environmental settings
environment_parameters = {
    "gravity": -10.0,
    "enable_wind": False,
    "wind_power": 5.0,
    "turbulence_power": 0.25
}

# --- Helper Function to Create Environment ---
def create_lunar_lander_env():
    return gym.make("LunarLander-v3",
                    continuous=False,
                    gravity=environment_parameters["gravity"],
                    enable_wind=environment_parameters["enable_wind"],
                    wind_power=environment_parameters["wind_power"],
                    turbulence_power=environment_parameters["turbulence_power"])

# --- Helper Function to Run a Single Training Iteration ---
def run_training_iteration(
    current_hyperparams: dict,
    iteration_name: str,
    current_folder:str,
    agent_type: str = "DQN" # Added a new parameter to specify agent type
):
    """
    Sets up and runs a single training session with the given hyperparameters.
    """
    print(f"\n--- Running Training: {iteration_name} ---")
    print(f"Hyperparameters for this run: {current_hyperparams}")
    print(f"Agent Type: {agent_type}") # Print agent type

    # re-seeding for each operation
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # Important for CUDA operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED) #for systems with multiple GPUs
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    log_dir_suffix = iteration_name.replace(" ", "_").replace(":", "_").replace(",", "").replace("[", "").replace("]", "").replace("__", "_")
    LOG_DIR = os.path.join(os.getcwd(), "logs", log_dir_suffix)
    CSV_FILENAME = os.path.join("tune/" + f"{current_folder}/" + f"{log_dir_suffix}.csv") 

    os.makedirs(LOG_DIR, exist_ok=True)

    env = create_lunar_lander_env()
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Initialize the correct DQNAgent or DDQNAgent based on agent_type
    if agent_type == "DDQN":
        agent = DDQNAgent(
            state_size=obs_dim,
            action_size=n_actions,
            hidden_layers=current_hyperparams["layers"],
            learning_rate=current_hyperparams["learning_rate"],
            gamma=current_hyperparams["gamma"],
            batch_size=current_hyperparams["batch_size"],
            target_update_freq=current_hyperparams["target_update_freq"]
        )
    else: # Default to DQN
        agent = DQNAgent(
            state_size=obs_dim,
            action_size=n_actions,
            hidden_layers=current_hyperparams["layers"],
            learning_rate=current_hyperparams["learning_rate"],
            gamma=current_hyperparams["gamma"],
            batch_size=current_hyperparams["batch_size"],
            target_update_freq=current_hyperparams["target_update_freq"]
        )
    
    writer = SummaryWriter(log_dir=LOG_DIR)

    # Pass the entire 'current_hyperparams' dictionary as the 'config' argument
    trainDQN(
        env,
        agent,
        writer,
        config=current_hyperparams, # Pass the dictionary here!
        csv_filename=CSV_FILENAME,
        iteration=iteration_name
    )

    writer.close()
    env.close() # Close environment after training
    print(f"--- Training {iteration_name} Finished ---")

# ==============================================================================
# --- MAIN TUNING SCRIPT ---
# ==============================================================================

if RUN_NOMINAL_BASELINE:
    print("\n##### Running Nominal Baseline Configuration #####")
    
    nominal_run_hyperparams = config.copy() 
    run_training_iteration(nominal_run_hyperparams, "nominal_baseline", "nominal")
    print("##### Nominal Baseline complete #####")

# ------------------------------------------------------------------------------

if TUNE_LAYERS:
    print("\n##### Tuning Network Layers (Architecture) #####")
    for layer_config in TUNING_RANGES["layers"]:
        current_run_hyperparams = config.copy()
        current_run_hyperparams["layers"] = layer_config
        layer_name = "_".join(map(str, layer_config))
        run_training_iteration(current_run_hyperparams, f"layers_{layer_name}", "layers")

    print("##### Layer Tuning complete #####")

# ------------------------------------------------------------------------------
# Here we select the layers that gave the best performance after evaluating the plot
config["layers"] = [128, 128]

if TUNE_BATCH_SIZE:
    print("\n##### Tuning Batch Size #####")
    for batch_size_value in TUNING_RANGES["batch_size"]:
        current_run_hyperparams = config.copy()
        current_run_hyperparams["batch_size"] = batch_size_value
        run_training_iteration(current_run_hyperparams, f"batch_size_{batch_size_value}", "batch")

    print("##### Batch Size Tuning complete #####")

# ------------------------------------------------------------------------------

config["batch_size"] = 64

if TUNE_LEARNING_RATE:

    print("\n##### Tuning Learning Rate (learning_rate) #####")
    for learning_rate_value in TUNING_RANGES["learning_rate"]:
        current_run_hyperparams = config.copy()
        current_run_hyperparams["learning_rate"] = learning_rate_value
        run_training_iteration(current_run_hyperparams, f"learning_rate_{learning_rate_value}", "learning_rate")
    
    print("##### Learning Rate Tuning complete #####")

# ------------------------------------------------------------------------------

config["learning_rate"] = 5e-4

if TUNE_EPSILON_DECAY:
    print("\n##### Tuning Epsilon Decay #####")
    for epsilon_decay_value in TUNING_RANGES["epsilon_decay"]:
        current_run_hyperparams = config.copy()
        current_run_hyperparams["epsilon_decay"] = epsilon_decay_value
        run_training_iteration(current_run_hyperparams, f"epsilon_decay_{epsilon_decay_value}", "epsilon_decay")

    print("##### Epsilon Decay Tuning complete #####")

# ------------------------------------------------------------------------------

config["epsilon_decay"] = 0.005

if TUNE_EPSILON_MIN: # Added this block
    print("\n##### Tuning Epsilon End #####")
    for epsilon_min_value in TUNING_RANGES["epsilon_min"]:
        current_run_hyperparams = config.copy()
        current_run_hyperparams["epsilon_min"] = epsilon_min_value
        run_training_iteration(current_run_hyperparams, f"epsilon_min_{epsilon_min_value}", "epsilon_min")

    print("##### Epsilon End Tuning complete #####")

# ------------------------------------------------------------------------------

if TUNE_GAMMA:
    print("\n##### Tuning Discount Factor #####")
    for gamma_value in TUNING_RANGES["gamma"]:
        current_run_hyperparams = config.copy()
        current_run_hyperparams["gamma"] = gamma_value
        run_training_iteration(current_run_hyperparams, f"gamma_{gamma_value}", "discount_factor")

    print("##### Discount Factor  Tuning complete #####")

config["gamma"] = 0.99

# ==============================================================================
# --- TEST WITH DDQN AGENT ---
# ==============================================================================
if RUN_DDQN_TEST:
    print("\n##### Running DDQN Agent Test with Final Hyperparameters #####")

    # Final hyperparameters you selected from your original tuning
    final_hyperparams = {
        "learning_rate": 5e-4,
        "batch_size": 64,
        "layers": [128, 128],
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.005,
        "gamma": 0.99,
        "target_update_freq": 1000,
        "max_episodes": 2000,
    }
    
    # Run a single training iteration with the DDQN agent
    run_training_iteration(
        current_hyperparams=final_hyperparams,
        iteration_name="ddqn_final_params",
        current_folder="ddqn_test",
        agent_type="DDQN" # Specify the agent type
    )

    print("##### DDQN Agent Test complete #####")

# ==============================================================================
# --- CHANGE OF ENVIRONMENT ---
# ==============================================================================

environment_parameters = {
    "gravity": -3.71,
    "enable_wind": True,
    "wind_power": 0.05,
    "turbulence_power": 0.05
}
if RUN_ENVIRONMENT:
    print("\n##### Running different environment configuration #####")
    
    run_hyperparams = config.copy() 
    run_training_iteration(run_hyperparams, "environment1", "environment")
    print("##### Environment complete #####")
