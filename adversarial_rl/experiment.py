""" This contains the code to attack an arbitrary trained agent on a gym env"""
import skeletor
import track
from cleverhans.attacks import FastGradientMethod
from cleverhans.loss import CrossEntropy, Loss

import baselines
from baselines.run import train
from baselines.common.vec_env import VecEnv
from baselines.common.tf_util import save_variables
from baselines.common.cmd_util import common_arg_parser

import os
import numpy as np

from skeletor.launcher import _add_default_args

_ATTACKS = {
    'fgsm': FastGradientMethod,
}

parser = None

def _load(args):
    setattr(args, 'num_timesteps', 0)
    extra_args = {}  # TODO see if we need this from baselines..run
    # try to load with pattern matching (this makes grid search easier)
    if args.load_path == '':
        args.load_path = './%s_%s.' % (args.alg, args.env)
    return train(args, extra_args)


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
            loss = CrossEntropy(model, attack=attack)
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
    global parser

    gym_parser = common_arg_parser()
    # now, we need to add the conflicting arguments to the gym parser
    options = ['--' + arg for arg in vars(args).keys()]
    for option in options:
        for action in gym_parser._actions:
            if vars(action)['option_strings'][0] == option:
                gym_parser._handle_conflict_resolve(None, [(option, action)])
    _add_args(gym_parser)
    _add_default_args(gym_parser)
    gym_args = gym_parser.parse_args()
    # vars(gym_args).update(vars(args))
    args = gym_args

    model, env = _load(args)
    episode_reward = eval_model(model, env, eval_steps=args.eval_steps,
                                attack_method=args.attack,
                                eps=args.eps)
    model_path = os.path.join(track.trial_dir(), 'model.ckpt')
    track.metric(trial_id=args.trial_id, reward=episode_reward,
                 model_path=model_path)


def _add_args(parser):
    # note: the baselines ddpg implementation is actually buggy, even 
    # out-of-the-box
    parser.add_argument('--alg', default='deepq', help='agent to train',
                        choices=['deepq', 'trpo_mpi', 'ppo2'])
    parser.add_argument('--env', default='PongNoFrameskip-v4',
                        help='Gym environment name to train on')
    parser.add_argument('--attack', default='fgsm',
                        choices=['fgsm'],
                        help='attack method to run')
    parser.add_argument('--attack-norm', default='l1',
                        choices=['l1', 'l2', 'l0'],
                        help="norm we use to constrain perturbation size")
    parser.add_argument('--eval_steps', default=10, type=int,
                        help='how many steps of the env to run')
    parser.add_argument('--eps', default=.1, type=float,
                        help='perturbation magnitude')
    parser.add_argument('--num-trials', default=10, type=int,
                        help='how many times to repeat the experiment')
    parser.add_argument('--load_path', default='', required=True,
                        help='Location of model with correct policy')
    # parser.add_argument('--train', action='store_const',
    #    # help='if true, just trains the plain model from scratch')




if __name__ == '__main__':
    skeletor.supply_args(_add_args)
    skeletor.execute(main)
