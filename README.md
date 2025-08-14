# Bio-Inspired 2025 – AE4350

## Setup Instructions

### 1. Create and activate a conda environment

```bash
conda create -n lunarlander-rl python=3.10
conda activate lunarlander-rl
```

### 2. Install the required packages

```bash
pip install -r requirements.txt
```

---

## Lunar Lander Environment Description

**State vector** (8 elements):

| Index | Description                |
| ----- | -------------------------- |
| 0     | Horizontal position (*x*)  |
| 1     | Vertical position (*y*)    |
| 2     | Horizontal velocity (*vx*) |
| 3     | Vertical velocity (*vy*)   |
| 4     | Lander angle (radians)     |
| 5     | Angular velocity           |
| 6     | Left leg contact (0 or 1)  |
| 7     | Right leg contact (0 or 1) |

**Actions** (discrete):

| Action ID | Description                   |
| --------- | ----------------------------- |
| 0         | Do nothing                    |
| 1         | Fire left orientation engine  |
| 2         | Fire main (downward) engine   |
| 3         | Fire right orientation engine |

---

## Code Overview

This repository contains scripts for training, tuning, and running a Deep Q-Network (DQN) agent to solve the Lunar Lander environment.

The tuning process follows a **sequential approach**:

1. Run a **nominal baseline** using literature-based hyperparameters to achieve approximately good performance.
2. Sequentially tune each selected hyperparameter while keeping the previously tuned ones fixed.
3. For each hyperparameter, evaluate different candidate values, select the best one, and proceed to the next parameter.

**Tuning order:**

1. Number of layers
2. Batch size
3. Learning rate
4. Epsilon decay rate
5. Minimum epsilon
6. Discount factor (*gamma*)

---

### Repository Contents

* **`train_DQN.py`**
  Contains all necessary functions to train the DQN agent.

* **`tune_DQN.py`**
  Sequentially tunes each selected hyperparameter.

  * Calls the `trainAgent` function from `train_DQN.py` for training.
  * Saves episode performance data as `.csv` files in the `tune/` directory, named using the tuned hyperparameter and its value.
  * Once the agent converges or the maximum number of episodes is reached, saves the trained model in `saved_models/` using the naming format:

    ```
    {hyperparameter}_{value}_ep{convergence_episode}.pth
    ```

* **`plot_tuned.py`**
  Reads the `.csv` results from tuning and plots them for comparison of different hyperparameter values.

* **`run_DQN.py`**
  Loads a saved trained model and lets the agent play the Lunar Lander game.

  * The model name must be manually specified from the `saved_models/` folder.

---

## Important Notes

* Tuning is controlled via **boolean flags** in the code. Set a flag to `True` to enable tuning for a given parameter.
* **Warning:** Running the tuning process with any flag enabled will **overwrite** the corresponding `.csv` file in the `tune/` directory.