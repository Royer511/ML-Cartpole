# ML-Cartpole: CartPole Q-Learning Project
### Project Overview
This project is an implementation of the Q-learning algorithm to train a reinforcement learning agent from OpenAI's Gym toolkit in the CartPole environment. The primary objective is to demonstrate the agent's ability to learn and adapt its strategy to maintain a pole in an upright position on a moving cart for as long as possible.

### Features
Q-Learning Algorithm: Implementation of a fundamental reinforcement learning algorithm.
State Space Discretization: Conversion of the CartPole's continuous state space into a discrete form for the Q-learning algorithm.
Hyperparameter Exploration: Experimentation with various hyperparameters to optimize the agent's learning process.
Progress Visualization: Graphical representation of the agent's learning over time, showcasing the impact of different learning rates and discount factors.
### Requirements
Python 3.x
OpenAI Gym
NumPy
Pandas
Matplotlib
tqdm (for progress bars)
### Installation
Install the required packages using pip:

pip install gym numpy pandas matplotlib tqdm

### Usage
To train the agent, execute the main script:
python cartpole.py

### Code Structure
CartPoleAgent: A class that defines the reinforcement learning agent.
train_agent: A function to train the agent with specified hyperparameters.
Visualization: Scripts for plotting the training results and conducting a comparative analysis of different hyperparameters.
Results
The agent's performance is evaluated based on average rewards and episode lengths under various hyperparameter settings. These metrics are visualized using line plots to illustrate the learning progress and the effectiveness of different parameter configurations.

**Contributing:**

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

**Contact:**

For any queries or suggestions, please get in touch with me at rroyer511@gmail.com
