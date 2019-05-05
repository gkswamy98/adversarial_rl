""" This contains the code to attack an arbitrary trained agent on a gym env"""
from baselines.common.vec_env import VecEnv
from baselines.common.cmd_util import common_arg_parser

from cleverhans.model import CallableModelWrapper

import gym
import numpy as np
import os

import skeletor
from skeletor.launcher import _add_default_args

import track

from adv.constants import ALG_LEARN_FNS, VALID_ALGS, POLICY_GRAD_ALGS, ATTACKS


def _debug_stats_str(stats, warn=False):
    """output: | mymetric1: 0 | mymetric2: 1 |"""
    all_floats = all(map(lambda f: isinstance(f, float), stats.values()))
    if not (all_floats or warn):
        track.debug("WARNING: I'm printing your metric arguments as floats")
    s = [' | %s: %.2f ' % (k, float(v)) for k, v in stats.items()]
    return ''.join(s)


def _load(alg, env_name, network, load_path):
    """
    Load the model we want to attack from a saved baselines checkpoint

    Returns:
    -------------------
    step function (arg: observation) that returns the best action
    env: the gym environment in which the agent will run
    y_placeholder: placeholder tensor for the network's output (action logits)
    obs_placeholder: placeholder for the network's input (observation)
    """
    if alg not in VALID_ALGS:
        raise ValueError("Unsupported alg: %s" % alg)

    # load the enivornment
    env = gym.make(env_name)

    learn = ALG_LEARN_FNS[alg]
    if alg in POLICY_GRAD_ALGS:
        pi = learn(env=env, network='mlp', total_timesteps=0,
                   load_path=load_path)
        # for these classes, we need to dig for the actual Model instance
        if alg == 'ppo2':
            pi = pi.act_model
        if alg == 'a2c':
            pi = pi.step_model

        def _act(observation, **kwargs):
            return pi.step(observation, **kwargs)[0]
        act = _act
        y_placeholder, obs_placeholder = pi.pi, pi.X
    else:
        # otherwise, we had to modify deepq to get this...
        assert alg == 'deepq', 'deepq is the only non-policy grad alg'
        act, y_placeholder, obs_placeholder = learn(env=env,
                                                    network=network,
                                                    total_timesteps=0,
                                                    load_path=load_path)
    return act, env, y_placeholder, obs_placeholder


def eval_model(model, env, y_placeholder, obs_placeholder, attack_method,
               attack_ord=2, num_rollouts=3, eps=0.1,
               trial_num=0, render=False):
    # cleverhans needs to get the logits tensor, but expects you to run
    # through and recompute it for the given observation
    # even though the graph is already created
    cleverhans_model = CallableModelWrapper(lambda o: y_placeholder, "logits")
    attack = ATTACKS[attack_method](cleverhans_model)

    fgsm_params = {'eps': eps, 'ord': attack_ord}

    # we'll keep tracking metrics here
    prev_done_step = 0
    stats = {}
    rewards = []

    stats['eval_step'] = 0
    stats['episode'] = 0
    stats['episode_reward'] = 0.
    stats['cumulative_reward'] = 0.

    obs = env.reset()
    num_episodes = 0
    while num_episodes < num_rollouts:
        # the attack_op tensor will generate the perturbed state!
        attack_op = attack.generate(obs_placeholder, **fgsm_params)
        adv_obs = attack_op.eval({obs_placeholder: obs[None, :]})
        action = model(adv_obs)[0]

        # it's time for my child to act out in this adversarial world
        obs, rew, done, _ = env.step(action)
        reward = rew[0] if isinstance(env, VecEnv) else rew
        if render:
            env.render()
        done = done.any() if isinstance(done, np.ndarray) else done

        # let's get our metrics
        stats['eval_step'] += 1
        stats['episode_reward'] += reward
        stats['cumulative_reward'] += reward
        stats['episode_len'] = stats['eval_step'] + prev_done_step

        if done:
            rewards.append(stats['episode_reward'])
            obs = env.reset()
            prev_done_step = stats['eval_step']
            track.debug("Finished episode %d! Stats: %s"
                        % (stats['episode'], _debug_stats_str(stats)))
            stats['episode'] += 1
            stats['episode_reward'] = 0
            stats['eval_step'] = 0
            num_episodes += 1
        # track metrics to access later through pandas
        track.metric(iteration=stats['eval_step'] + prev_done_step,
                     trial_num=trial_num,
                     **stats)

    env.close()
    print('REWARDS', rewards)
    return stats  # gimme the final stats for the episode


def _fix_baseline_args(args):
    """
    Unfortunately, baselines has its own parser that will always run
    if you try to use it as a separate module. We have to integrate
    our args with theirs here.
    """
    gym_parser = common_arg_parser()
    options = ['--' + arg for arg in vars(args).keys()]
    for option in options:
        for action in gym_parser._actions:
            if vars(action)['option_strings'][0] == option:
                gym_parser._handle_conflict_resolve(None, [(option, action)])
    _add_args(gym_parser)
    _add_default_args(gym_parser)
    gym_parser = gym_parser.parse_args()
    vars(args).update(vars(gym_parser))
    return args


def main(args):
    args = _fix_baseline_args(args)

    # try to load with pattern matching (this makes grid search easier)
    if args.load_path == '':
        default_path = os.path.join(args.model_dir, '%s_%s.pkl'
                                    % (args.alg, args.env))
        track.debug("Didn't find a load_path, so we will try to load from: %s"
                    % default_path)
        load_path = default_path

    model, env, y_placeholder, obs_placeholder = _load(
        args.alg, args.env, args.network, load_path)

    final_stats = eval_model(model, env, y_placeholder, obs_placeholder,
                             num_rollouts=args.num_rollouts,
                             attack_method=args.attack,
                             attack_ord=args.attack_ord,
                             eps=args.eps,
                             render=args.render)
    track.debug("FINAL STATS:%s" % _debug_stats_str(final_stats))


def _add_args(parser):
    parser.add_argument('--alg', default='deepq', help='agent to train',
                        choices=list(ALG_LEARN_FNS.keys()))
    parser.add_argument('--env', default='CartPole-v0',
                        help='Gym environment name to train on')
    parser.add_argument('--attack', default='fgsm',
                        choices=list(ATTACKS.keys()),
                        help='attack method to run')
    parser.add_argument('--network', default='mlp', type=str,
                        help='policy network arhitecture')
    parser.add_argument('--attack_ord', default=2, type=int,
                        help="norm we use to constrain perturbation size")
    parser.add_argument('--num_rollouts', default=10, type=int,
                        help='how many episodes to run for each attack')
    parser.add_argument('--eps', default=.1, type=float,
                        help='perturbation magnitude')
    parser.add_argument('--model_dir', default='./models', type=str,
                        help='where to look for model pkls by default')
    parser.add_argument('--load_path', default='',
                        help='location of model .pkl with correct policy')
    parser.add_argument('--render', action='store_true',
                        help='if true, render the actual gym env on-screen')


if __name__ == '__main__':
    skeletor.supply_args(_add_args)
    skeletor.execute(main)
