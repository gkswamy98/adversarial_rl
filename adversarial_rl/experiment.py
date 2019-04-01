""" This contains the code to attack an arbitrary trained agent on a gym env"""
import track
from cleverhans.attacks import FastGradientMethod
from cleverhans.loss import CrossEntropyLoss, Loss

import baselines
from baselines.common.vec_env import VecEnv
from baselines.common.tf_util import save_variables

import os
import numpy as np

_ATTACKS = {
    'fgsm': FastGradientMethod,
}


def _load(args):
    setattr(args, 'num_timesteps', 0)
    extra_args = {}  # TODO see if we need this from baselines..run
    # try to load with pattern matching (this makes grid search easier)
    if args.load_path == '':
        args.load_path = './%s_%s' % (args.alg, args.env)
    return baselines.train(args, extra_args)


class WrappedModel:
    """ cleverhans expected a get_logits function for the model """
    def __init__(self, model):
        self.model = model

    def get_logits(x, **kwargs):
        # get logits from the model for this action ....
        pass


def eval_model(model, env, attack_method, eval_steps=10, **attack_params):
    obs = env.reset()
    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))

    cleverhans_model = WrappedModel(model)
    attack = _ATTACKS[attack_method](model)

    episode_rew = 0
    for _ in range(eval_steps):
        if state is not None:
            # perform the attack!
            loss = CrossEntropyLoss(model, attack=attack)
            logits = cleverhans_model.get_logits(state)
            adv_state = attack.generate(state, **attack_params)

            actions, _, state, _ = model.step(obs, S=adv_state, M=dones)
        else:
            actions, _, _, _ = model.step(obs)

        obs, rew, done, _ = env.step(actions)
        episode_rew += rew[0] if isinstance(env, VecEnv) else rew
        env.render()
        done = done.any() if isinstance(done, np.ndarray) else done
        if done:
            print('episode_rew={}'.format(episode_rew))
            episode_rew = 0
            obs = env.reset()
    env.close()
    return episode_rew


def main(args):
    if args.train:
        model = baselines.main(args)
        fname = './%s_%s' % (args.alg, args.env)
        save_variables(fname)
        return

    model, env = _load(args)
    episode_reward = eval_model(model, env, eval_steps=args.eval_steps,
                                attack=args.attack,
                                eps=args.eps)
    model_path = os.path.join(track.trial_dir(), 'model.ckpt')
    track.metric(trial_id=args.trial_id, reward=episode_reward,
                 model_path=model_path)
