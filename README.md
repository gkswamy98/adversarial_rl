# Adversarial Attacks on Deep Reinforcement Learning

## Installation
1) Run `git clone --recurse-submodules https://github.com/gkswamy98/adversarial_rl.git` to clone with submodules and `cd adversarial_rl`.
2) Run `pip install -e .` to install the main dependencies.
3) Run `mv baselines ../baselines`, `cd ../baselines`, then `pip install -e . --user` to install our baselines fork.

## Running Experiments
* To train models, run `bash scripts/train.sh`. 
  * To parallelize training runs, modify the training script to echo commands instead of executing them and run `bash ../scripts/train.sh | xargs -PN -ICMD /bin/bash -exc CMD`, replacing N with the number of cores available.
* To attack trained models, run `bash scripts/attack.sh`.
 * Use the same trick as above to parallelize.

## Results
PPO, TRPO, A2C and DQN models trained on CartPole and Acrobot are in the `models/` folder. Results of attacks are available in accompanying writeup.

## Authors
* [@gkswamy98](https://github.com/gkswamy98)
* [@noahgolmant](https://github.com/noahgolmant)
