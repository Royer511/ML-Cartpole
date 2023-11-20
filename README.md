# ML-Cartpole
CartPole Q-Learning Project
**Project Overview:**

This project uses the Q-learning algorithm to train a reinforcement learning agent to balance a pole on a moving cart. The environment used for this project is the CartPole environment from OpenAI's Gym toolkit. The goal is to demonstrate the agent's ability to learn and adapt its strategy to maintain the pole upright for as long as possible.

**Features:**

Implementation of the Q-learning algorithm.
Discretization of the continuous state space of the CartPole environment.
Exploration of various hyperparameters to optimize the learning process.
Visualization of the agent's learning progress over time.

**Requirements:**

Python 3.x
OpenAI Gym
NumPy
Pandas
Matplotlib
tqdm (for progress bar)

**You can install these packages using pip:**

pip install gym numpy pandas matplotlib tqdm

**Usage:**

To train the agent, run the main script:
python cartpole.py

**Code Structure:**

CartPoleAgent: Class defining the reinforcement learning agent.
train_agent: Function to train the agent with specified hyperparameters.
Visualization: Code to plot the training results and comparative analysis of hyperparameters.

**Results:**

The agent's performance is evaluated based on the average rewards and episode lengths across different hyperparameter settings. The results are visualized using line plots to show the agent's learning progress and the impact of different learning rates and discount factors.

**Contributing:**

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

**Contact:**

For any queries or suggestions, please get in touch with me at rroyer511@gmail.com
