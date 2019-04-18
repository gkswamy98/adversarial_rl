""" This contains the code to attack an arbitrary trained agent on a gym env"""
from baselines.common.vec_env import VecEnv
from baselines.common.cmd_util import common_arg_parser
from baselines import deepq, trpo_mpi, ppo2, a2c  # home of modified learn fns 

from cleverhans.model import CallableModelWrapper
from cleverhans.attacks import FastGradientMethod

import gym
import numpy as np

import skeletor
from skeletor.launcher import _add_default_args

import track

_ATTACKS = {
    'fgsm': FastGradientMethod,
}

# unfortunately, the  baselines ddpg implementation is too buggy to use,
# even out-of-the-box
_ALG_LEARN_FNS = {
    'deepq': deepq.learn,
    'trpo_mpi': trpo_mpi.learn,
    'ppo2': ppo2.learn,
    'a2c': a2c.learn
}

_VALID_ALGS = list(_ALG_LEARN_FNS.keys())

_POLICY_GRAD_ALGS = [
    'trpo_mpi',
    'ppo2',
    'a2c'
]


def _debug_stats_str(stats):
    " | mymetric1: 0 | mymetric2: 1 |"
    assert all(map(lambda f: isinstance(f, float), stats.values())),\
        "I'm lazy and expect all tracked metrics to be floats"
    return sum([' | %s: %.2f ' % (k, v) for k, v in stats.items()]) + '|'


def _load(args):
    """
    Load the model we want to attack from a saved baselines checkpoint

    Returns:
    -------------------
    step function (arg: observation) that returns the best action
    env: the gym environment in which the agent will run
    y_placeholder: placeholder tensor for the network's output (action logits)
    obs_placeholder: placeholder for the network's input (observation)
    """
    if args.alg not in _VALID_ALGS:
        raise ValueError("Unsupported alg: %s" % args.alg)
    # try to load with pattern matching (this makes grid search easier)
    if args.load_path == '':
        default_path = './%s_%s.pkl' % (args.alg, args.env)
        track.debug("Didn't find a load_path, so we will try to load from: %s"
                    % default_path)
        args.load_path = default_path

    # load the enivornment
    env = gym.make(args.env)

    learn = _ALG_LEARN_FNS[args.alg]
    if args.alg in _POLICY_GRAD_ALGS:
        pi = learn(env=env, network='mlp', total_timesteps=0,
                   load_path=args.load_path)
        # for these classes, we need to dig for the actual Model instance
        if args.alg == 'ppo2':
            pi = pi.act_model
        if args.alg == 'a2c':
            pi = pi.step_model

        def _act(pi, observation, **kwargs):
            return pi.step(observation, **kwargs)[0]
        y_placeholder, obs_placeholder = pi.pi, pi.X
    else:
        assert args.alg == 'deepq', 'deepq is the only non-policy grad alg'
        act, y_placeholder, obs_placeholder = learn(env=env,
                                                    network=args.network,
                                                    total_timesteps=0,
                                                    load_path=args.load_path)
    return act, env, y_placeholder, obs_placeholder


def eval_model(model, env, q_placeholder, obs_placeholder, attack_method,
               eval_steps=1000, eps=0.1, trial_num=0):
    # cleverhans needs to get the logits tensor, but expects you to run
    # through and recompute it for the given observation
    # even tho the graph is already created
    cleverhans_model = CallableModelWrapper(lambda o: q_placeholder, "logits")
    attack = _ATTACKS[attack_method](cleverhans_model)
    fgsm_params = {'eps': eps}

    # we'll keep tracking metrics here
    cumulative_reward = 0.
    episode_reward = 0.
    prev_done_step = 0
    episode = 0

    obs = env.reset()
    for i in range(eval_steps):
        # the attack_op tensor will generate the perturbed state!
        attack_op = attack.generate(obs_placeholder, **fgsm_params)
        adv_obs = attack_op.eval({obs_placeholder: obs[None, :]})
        action = model(adv_obs)[0]

        # it's time for my child to act out in this adversarial world
        obs, rew, done, _ = env.step(action)
        reward = rew[0] if isinstance(env, VecEnv) else rew
        env.render()
        done = done.any() if isinstance(done, np.ndarray) else done

        # let's get our metrics
        episode_reward += reward
        cumulative_reward += reward
        episode_len = i - prev_done_step
        prev_done_step = i

        if done:
            obs = env.reset()
            stats = {
                'eval_step': i,
                'episode_len': episode_len,
                'episode_reward': episode_reward,
                'cumulative_reward': cumulative_reward,
                'episode': episode
            }
            episode += 1
            episode_reward = 0

            track.metric(iteration=i, trial_num=trial_num,
                         **stats)
            track.debug("Finished episode %d! Stats: %s"
                        % (episode, _debug_stats_str(stats)))
    env.close()
    return stats  # gimme the final stats for the episode


def main(args):
    gym_parser = common_arg_parser()
    # now, we need to add the conflicting arguments to the gym parser
    # (this is a bit messy)
    options = ['--' + arg for arg in vars(args).keys()]
    for option in options:
        for action in gym_parser._actions:
            if vars(action)['option_strings'][0] == option:
                gym_parser._handle_conflict_resolve(None, [(option, action)])
    _add_args(gym_parser)
    _add_default_args(gym_parser)
    args = gym_parser.parse_args()

    model, env, y_placeholder, obs_placeholder = _load(args)
    final_stats = eval_model(model, env, y_placeholder, obs_placeholder,
                             eval_steps=args.eval_steps,
                             attack_method=args.attack,
                             eps=args.eps)
    track.debug("FINAL STATS: %s" % _debug_stats_str(final_stats))


def _add_args(parser):
    parser.add_argument('--alg', default='deepq', help='agent to train',
                        choices=['deepq', 'trpo_mpi', 'ppo2', 'a2c'])
    parser.add_argument('--env', default='PongNoFrameskip-v4',
                        help='Gym environment name to train on')
    parser.add_argument('--attack', default='fgsm',
                        choices=['fgsm'],
                        help='attack method to run')
    parser.add_argument('--network', default='mlp', type=str,
                        help='policy network arhitecture')
    parser.add_argument('--attack-norm', default='l1',
                        choices=['l1', 'l2', 'l0'],
                        help="norm we use to constrain perturbation size")
    parser.add_argument('--eval_steps', default=10, type=int,
                        help='how many steps of the env to run')
    parser.add_argument('--eps', default=.1, type=float,
                        help='perturbation magnitude')
    parser.add_argument('--num_trials', default=10, type=int,
                        help='how many times to repeat the experiment')
    parser.add_argument('--load_path', default='', required=True,
                        help='Location of model with correct policy')


if __name__ == '__main__':
    skeletor.supply_args(_add_args)
    skeletor.execute(main)
