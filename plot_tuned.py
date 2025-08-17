import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import csv

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 16,
    "font.size": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# --- Configuration Flags (from your tune_DQN.py) ---
# Ensure these match your tuning script's enabled flags
PLOT_NOMINAL_BASELINE = False
PLOT_LAYERS = True
PLOT_BATCH_SIZE = True
PLOT_LEARNING_RATE = True
PLOT_EPSILON_DECAY = True
PLOT_EPSILON_MIN = True
PLOT_GAMMA = True
PLOT_ENVIRONMENT = True 


# --- Base directory for saving CSVs ---
BASE_TUNE_DIR = "tune"
os.makedirs("plots", exist_ok=True)

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

# --- General Plotting Function ---
def plot_tuning_results(
    param_name: str,
    param_values: list,
    folder_name: str,
    nominal_baseline_csv_path: str,
    window_size: int = 100, # Window for calculating rolling average
    plot_nominal_baseline: bool = True
):
    plt.figure(figsize=(10, 4))
    plt.xlabel('Episode [-]', fontsize=18)
    plt.ylabel(f'Mean Reward [-]', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.6)

    convergence_summary = {}

    if plot_nominal_baseline and os.path.exists(nominal_baseline_csv_path):
        try:
            df_nominal = pd.read_csv(nominal_baseline_csv_path, comment='#') # Use comment='# for new header
            df_nominal['rolling_mean_reward'] = df_nominal['reward'].rolling(window=window_size).mean()
            plt.plot(df_nominal['episode'], df_nominal['rolling_mean_reward'],
                     label='Nominal', linestyle='--', color='black', alpha=0.8)
            # Find the last valid rolling mean reward for the nominal baseline
            last_nominal_reward = df_nominal['rolling_mean_reward'].dropna().iloc[-1]
            convergence_summary["Nominal Baseline"] = f"{last_nominal_reward:.2f}"
        except Exception as e:
            print(f"Error reading nominal baseline CSV {nominal_baseline_csv_path}: {e}")
            plot_nominal_baseline = False # Don't try to plot it if there's an error

    for value in param_values:
        if param_name == "layers":
            # For layers, the name in the CSV is "layers_128_64" etc.
            sanitized_value_name = "_".join(map(str, value))
            csv_filename = f"{param_name}_{sanitized_value_name}.csv"
            label = f"{value}"
        else:
            csv_filename = f"{param_name}_{value}.csv"
            label = f"{value}"

        file_path = os.path.join(BASE_TUNE_DIR, folder_name, csv_filename)

        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, comment='#') # Use comment='#' to skip your hyperparams line
                df['rolling_mean_reward'] = df['reward'].rolling(window=window_size).mean()
                plt.plot(df['episode'], df['rolling_mean_reward'], label=label)

                # Get the last non-NaN rolling mean reward
                last_rolling_mean = df['rolling_mean_reward'].dropna().iloc[-1]
                convergence_summary[label] = f"{last_rolling_mean:.2f}"

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

    plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0), fontsize=12)
    plt.tick_params(axis='y', labelsize=18)
    plt.tick_params(axis='x', labelsize=18)
    plt.gca().yaxis.get_offset_text().set_fontsize(18)

    plt.subplots_adjust(left=0.13, right=0.95, top=0.95, bottom=0.2)
    plt.xlim([100,2000])
    plt.savefig(f"plots/{param_name}.pdf")
    
    print(f"\n--- Final Rolling Mean Rewards for {param_name} Tuning ---")
    for param_config, reward in convergence_summary.items():
        print(f"{param_config}: {reward}")
    print("-" * 50)

def plot_combined_tuning_results(
    param_name: str,
    param_values: list,
    folder_name: str,
    nominal_baseline_csv_path: str,
    window_size: int = 100,
    plot_nominal_baseline: bool = True
):
    # This will dynamically find the max episode value to set the x-axis limits.
    max_episode = 0

    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1])

    # --- Left Subplot: Episodes vs. Mean Reward (Line Plot) ---
    ax_line = fig.add_subplot(gs[0, 0])
    ax_line.set_xlabel('Episode [-]', fontsize=18)
    ax_line.set_ylabel(f'Mean Reward [-]', fontsize=18)
    ax_line.grid(True, linestyle='--', alpha=0.6)

    convergence_summary = {} # To still print summary if needed

    if plot_nominal_baseline and os.path.exists(nominal_baseline_csv_path):
        try:
            df_nominal = pd.read_csv(nominal_baseline_csv_path, comment='#')
            df_nominal['rolling_mean_reward'] = df_nominal['reward'].rolling(window=window_size).mean()
            ax_line.plot(df_nominal['episode'], df_nominal['rolling_mean_reward'],
                     label='Nominal Baseline', linestyle='--', color='black', alpha=0.8)
            last_nominal_reward = df_nominal['rolling_mean_reward'].dropna().iloc[-1]
            convergence_summary["Nominal Baseline"] = f"{last_nominal_reward:.2f}"
            
            # Update max_episode with the last episode from the nominal baseline
            max_episode = max(max_episode, df_nominal['episode'].max())

        except Exception as e:
            print(f"Error reading nominal baseline CSV {nominal_baseline_csv_path}: {e}")
            plot_nominal_baseline = False

    for value in param_values: # Iterate directly over param_values to preserve order
        if param_name == "layers":
            sanitized_value_name = "_".join(map(str, value))
            csv_filename = f"{param_name}_{sanitized_value_name}.csv"
            label = f"{value}"
        else:
            csv_filename = f"{param_name}_{value}.csv"
            label = f"{param_name}: {value}"

        file_path = os.path.join(BASE_TUNE_DIR, folder_name, csv_filename)

        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, comment='#')
                df['rolling_mean_reward'] = df['reward'].rolling(window=window_size).mean()
                ax_line.plot(df['episode'], df['rolling_mean_reward'], label=label)
                last_rolling_mean = df['rolling_mean_reward'].dropna().iloc[-1]
                convergence_summary[label] = f"{last_rolling_mean:.2f}"
                
                # Update max_episode with the last episode from the current dataframe
                max_episode = max(max_episode, df['episode'].max())

            except Exception as e:
                print(f"Error processing {file_path} for line plot: {e}")
        else:
            print(f"File not found: {file_path}")
    
    # Set x-axis limit based on the maximum episode found
    if max_episode > 0:
        ax_line.set_xlim([100, max_episode]) 
    else:
        ax_line.set_xlim([100, 2000]) # Fallback to original if no data is found

    ax_line.legend(loc='lower right', bbox_to_anchor=(1.0, 0), fontsize=14) # Smaller legend for subplot
    ax_line.tick_params(axis='y', labelsize=18) # Smaller tick labels for subplot
    ax_line.tick_params(axis='x', labelsize=18)
    ax_line.yaxis.get_offset_text().set_fontsize(18)

    # --- Right Subplot: Parameters vs. Final Mean Reward (Bar Chart) ---
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_bar.set_xlabel(param_name.replace('_', ' ').title(), fontsize=18)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.7) # Grid only on y-axis

    final_rewards = []
    bar_labels = [] # Use a separate variable to avoid conflict with line plot labels

    for value in param_values: # Iterate directly over param_values to preserve order
        if param_name == "layers":
            sanitized_value_name = "_".join(map(str, value))
            csv_filename = f"{param_name}_{sanitized_value_name}.csv"
            display_label = f"{'/'.join(map(str, value))}"
        else:
            csv_filename = f"{param_name}_{value}.csv"
            display_label = str(value)

        file_path = os.path.join(BASE_TUNE_DIR, folder_name, csv_filename)

        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, comment='#')
                rolling_mean = df['reward'].rolling(window=window_size).mean()
                final_reward = rolling_mean.dropna().iloc[-1]
                final_rewards.append(final_reward)
                bar_labels.append(display_label)
            except Exception as e:
                print(f"Error processing {file_path} for bar chart: {e}")
        else:
            print(f"File not found for bar chart: {file_path}")

    if final_rewards: # Only plot if there's data
        bars = ax_bar.bar(bar_labels, final_rewards, color='skyblue')
        ax_bar.axhline(0, color='gray', linewidth=0.8)

        for bar in bars:
            yval = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2, 5 if yval >= 0 else -10,
                     round(yval, 2), ha='center', va='bottom' if yval >=0 else 'top', fontsize=14)
            
    ax_bar.tick_params(axis='y', labelsize=18) # Smaller tick labels for subplot
    ax_bar.tick_params(axis='x', labelsize=18, rotation=45)
    ax_bar.yaxis.get_offset_text().set_fontsize(18)

    plt.tight_layout() # Adjust layout to prevent overlapping
    plt.savefig(f"plots/{param_name}_combined.pdf")
    plt.close(fig) # Close the figure using the figure object

    # You can still print the convergence summary here if you want:
    print(f"\n--- Final Rolling Mean Rewards for {param_name} Tuning ---")
    for param_config, reward in convergence_summary.items():
        print(f"{param_config}: {reward}")
    print("-" * 50)



# --- Define paths for plotting ---
NOMINAL_BASELINE_CSV = os.path.join(BASE_TUNE_DIR, "nominal", "nominal_baseline.csv")

# ==============================================================================
# --- PLOTTING SCRIPT ---
# ==============================================================================

if PLOT_NOMINAL_BASELINE:
    print("\n##### Plotting Nominal Baseline Results #####")
    CSV_FILE = "tune/nominal/nominal_baseline.csv"  # Change path if needed
    WINDOW = 100  # Moving average window size

    # === READ DATA ===
    episodes = []
    rewards = []

    with open(CSV_FILE, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:  # skip empty lines
                if row[0] == "episode":  # skip header
                    continue
                ep, rew = row
                episodes.append(int(ep))
                rewards.append(float(rew))

    # === PLOT RAW REWARDS ===
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, rewards, label="Episode Reward", alpha = 0.9, color="grey")
    plt.xlabel('Episode [-]', fontsize=18)
    plt.ylabel(f'Mean Reward [-]', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.6)

    # === PLOT MOVING AVERAGE ===
    if len(rewards) >= WINDOW:
        moving_avg = np.convolve(rewards, np.ones(WINDOW)/WINDOW, mode='valid')
        plt.plot(episodes[WINDOW-1:], moving_avg, label=f"Moving Avg ({WINDOW})", color='tab:red', linewidth=2)

    plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0), fontsize=12)
    plt.tick_params(axis='y', labelsize=18)
    plt.tick_params(axis='x', labelsize=18)
    plt.gca().yaxis.get_offset_text().set_fontsize(18)

    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.2)
    plt.xlim([100,588])
    plt.savefig(f"plots/nominal_baseline.pdf")


if PLOT_LAYERS:
    print("\n##### Plotting Layers Tuning Results #####")
    plot_tuning_results(
        param_name="layers",
        param_values=TUNING_RANGES["layers"],
        folder_name="layers",
        nominal_baseline_csv_path=NOMINAL_BASELINE_CSV,
        plot_nominal_baseline=False
    )
    plot_combined_tuning_results(
        param_name="layers",
        param_values=TUNING_RANGES["layers"],
        folder_name="layers",
        nominal_baseline_csv_path=NOMINAL_BASELINE_CSV,
        plot_nominal_baseline=False
    )


if PLOT_LEARNING_RATE:
    print("\n##### Plotting Learning Rate Tuning Results #####")
    plot_tuning_results(
        param_name="learning_rate",
        param_values=TUNING_RANGES["learning_rate"],
        folder_name="learning_rate",
        nominal_baseline_csv_path=NOMINAL_BASELINE_CSV,
        plot_nominal_baseline=True # Include nominal baseline in this plot
    )
    plot_combined_tuning_results(
        param_name="learning_rate",
        param_values=TUNING_RANGES["learning_rate"],
        folder_name="learning_rate",
        nominal_baseline_csv_path=NOMINAL_BASELINE_CSV,
        plot_nominal_baseline=True # Include nominal baseline in this plo
    )

if PLOT_BATCH_SIZE:
    print("\n##### Plotting Batch Size Tuning Results #####")
    plot_tuning_results(
        param_name="batch_size",
        param_values=TUNING_RANGES["batch_size"],
        folder_name="batch",
        nominal_baseline_csv_path=NOMINAL_BASELINE_CSV,
        plot_nominal_baseline=True
    )
    plot_combined_tuning_results(
        param_name="batch_size",
        param_values=TUNING_RANGES["batch_size"],
        folder_name="batch",
        nominal_baseline_csv_path=NOMINAL_BASELINE_CSV,
        plot_nominal_baseline=True
    )

if PLOT_EPSILON_DECAY:
    print("\n##### Plotting Epsilon Decay Tuning Results #####")
    plot_tuning_results(
        param_name="epsilon_decay",
        param_values=TUNING_RANGES["epsilon_decay"],
        folder_name="epsilon_decay",
        nominal_baseline_csv_path=NOMINAL_BASELINE_CSV,
        plot_nominal_baseline=True
    )
    plot_combined_tuning_results(
        param_name="epsilon_decay",
        param_values=TUNING_RANGES["epsilon_decay"],
        folder_name="epsilon_decay",
        nominal_baseline_csv_path=NOMINAL_BASELINE_CSV,
        plot_nominal_baseline=True
    )

if PLOT_EPSILON_MIN:
    print("\n##### Plotting Epsilon Min Tuning Results #####")
    plot_tuning_results(
        param_name="epsilon_min",
        param_values=TUNING_RANGES["epsilon_min"],
        folder_name="epsilon_min",
        nominal_baseline_csv_path=NOMINAL_BASELINE_CSV,
        plot_nominal_baseline=True
    )
    plot_combined_tuning_results(
        param_name="epsilon_min",
        param_values=TUNING_RANGES["epsilon_min"],
        folder_name="epsilon_min",
        nominal_baseline_csv_path=NOMINAL_BASELINE_CSV,
        plot_nominal_baseline=True
    )


if PLOT_GAMMA:
    print("\n##### Plotting Gamma Tuning Results #####")
    plot_tuning_results(
        param_name="gamma",
        param_values=TUNING_RANGES["gamma"],
        folder_name="discount_factor", # Note: your folder name is "discount_factor"
        nominal_baseline_csv_path=NOMINAL_BASELINE_CSV,
        plot_nominal_baseline=True
    )
    plot_combined_tuning_results(
        param_name="gamma",
        param_values=TUNING_RANGES["gamma"],
        folder_name="discount_factor", # Note: your folder name is "discount_factor"
        nominal_baseline_csv_path=NOMINAL_BASELINE_CSV,
        plot_nominal_baseline=True
    )

print("\nAll selected plotting phases completed.")

# plot different environment configurations

if PLOT_ENVIRONMENT:
        
    # --- Constants and Directory Setup ---
    BASE_TUNE_DIR = "tune"
    NOMINAL_BASELINE_PATH = os.path.join(BASE_TUNE_DIR, "nominal", "nominal_baseline.csv")
    WINDOW_SIZE = 100 # Window for calculating rolling average

    # List of environment files to plot
    ENVIRONMENT_FILES = [
        "environment.csv",
        "environment1.csv",
        "environment2.csv",
        "environment3.csv"
    ]
    ENVIRONMENT_FOLDER = os.path.join(BASE_TUNE_DIR, "environment")

    # To dynamically set the x-axis limits
    max_episode = 0
    convergence_summary = {}

    plt.figure(figsize=(10, 4))
    plt.xlabel('Episode [-]', fontsize=18)
    plt.ylabel(f'Mean Reward [-]', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.6)

    # --- Plot Each Environment ---
    for i, filename in enumerate(ENVIRONMENT_FILES, 1):
        file_path = os.path.join(ENVIRONMENT_FOLDER, filename)
        label = f"Environment {i}"

        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, comment='#')
                df['rolling_mean_reward'] = df['reward'].rolling(window=WINDOW_SIZE).mean()
                plt.plot(df['episode'], df['rolling_mean_reward'], label=label)

                last_rolling_mean = df['rolling_mean_reward'].dropna().iloc[-1]
                convergence_summary[label] = f"{last_rolling_mean:.2f}"
                max_episode = max(max_episode, df['episode'].max())

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

    # --- Final Plot Customization ---
    plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0), fontsize=12)
    plt.tick_params(axis='y', labelsize=18)
    plt.tick_params(axis='x', labelsize=18)
    plt.gca().yaxis.get_offset_text().set_fontsize(18)
    plt.subplots_adjust(left=0.13, right=0.95, top=0.95, bottom=0.2)

    if max_episode > 0:
        plt.xlim([100, max_episode])
    else:
        # Fallback if no data is found
        plt.xlim([100, 2000])

    # --- Save Plot and Print Summary ---
    plt.savefig("plots/environment_comparison.pdf")
    plt.close()

    print(f"\n--- Final Rolling Mean Rewards for Environment Tuning ---")
    for env_config, reward in convergence_summary.items():
        print(f"{env_config}: {reward}")
    print("-" * 50)