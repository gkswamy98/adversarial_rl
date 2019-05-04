# Reproducing "Adversarial Attacks on Neural Network Policies"

## Installation

Run `pip install -e .` to install the main dependencies.

Unfortunately, the baselines repo requires separate installation. We had to make our own fork to make these attacks work. Please follow the installation instructions in [this repository](https://github.com/noahgolmant/baselines).


## CartPole-v0

Alg  | Timesteps
------------
TRPO  | 1e5
PPO   | 3e6
DEEPQ | 2e6
A2C   | 3e6
-------------


For each (alg, env) pair, 5 trials, 5 epsilon values
track average reward
