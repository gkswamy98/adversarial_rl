# Adversarial Attacks on Deep Reinforcement Learning

## Installation
1) Run `pip install -e .` to install the main dependencies.
2) Follow the instructions at [this link](https://github.com/noahgolmant/baselines) to install our baselines fork.

## Running Experiments
* To train models, run `cd baselines` and then `bash ../scripts/train.sh`. 
  * To parallelize training runs, run `bash ../scripts/train.sh | xargs -PN -ICMD /bin/bash -exc CMD`, replacing N with the number of cores available.
* To attack trained models, TODO

## Results
PPO, TRPO, A2C and DQN models trained on CartPole and Acrobot are in the `models/` folder. Results of attacks are available in accompanying writeup.

## Authors
* [@gkswamy98](https://github.com/gkswamy98)
* [@noahgolmant](https://github.com/noahgolmant)
