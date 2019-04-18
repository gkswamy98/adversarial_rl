""" This contains the code to attack an arbitrary trained agent on a gym env"""
import gym

from baselines import deepq, trpo_mpi, ppo2, a2c

from baselines.common.vec_env import VecEnv
from baselines.common.cmd_util import common_arg_parser

from cleverhans.model import CallableModelWrapper
from cleverhans.attacks import FastGradientMethod
from cleverhans.loss import CrossEntropy, Loss

import skeletor
import track

import numpy as np


from skeletor.launcher import _add_default_args

_ATTACKS = {
    'fgsm': FastGradientMethod,
}

parser = None

def _load(args):
    # try to load with pattern matching (this makes grid search easier)
    if args.load_path == '':
        args.load_path = './%s_%s.pkl' % (args.alg, args.env)

    # load the enivornment
    env = gym.make(args.env)

    if args.alg == 'deepq':
        act, q_placeholder, obs_placeholder = deepq.learn(env=env,
                                                          network='mlp',
                                                          total_timesteps=0,
                                                          load_path=args.load_path)
        return act, env, q_placeholder, obs_placeholder
    elif args.alg == 'trpo_mpi':
        pi = trpo_mpi.learn(env=env,
                            network='mlp',
                            total_timesteps=0,
                            load_path=args.load_path)
    elif args.alg == 'ppo2':
        pi = ppo2.learn(env=env,
                        network='mlp',
                        total_timesteps=0,
                        load_path=args.load_path).act_model
    elif args.alg == 'a2c':
        pi = a2c.learn(env=env,
                        network='mlp',
                        total_timesteps=0,
                        load_path=args.load_path).step_model
    else:
        print("u fucked up")

    def _step(observation, **extra_feed):
        # step returns a, v, state, neglogp
        return pi.step(observation, **extra_feed)[0]
    return _step, env, pi.pi, pi.X



class WrappedModel(CallableModelWrapper):
    def __init__(self, forward, name):
        super(WrappedModel, self).__init__(forward, name)
        self.forward = forward

    def get_logits(self, x):
        return self.forward(x)


def eval_model(model, env, q_placeholder, obs_placeholder, attack_method,
               eval_steps=1000, eps=0.1):
    obs = env.reset()

    # functional callable model wrapper for cleverhans needs a fn, not object
    cleverhans_model = WrappedModel(lambda o: q_placeholder, "logits")
    attack = _ATTACKS[attack_method](cleverhans_model)

    episode_rew = 0
    fgsm_params = {
          'eps': eps,
      }
    for _ in range(eval_steps):
        # perform the attack!
        loss = CrossEntropy(cleverhans_model, attack=attack)
        logits = cleverhans_model.get_logits(obs_placeholder)
        # import pdb; pdb.set_trace()
        attack_op = attack.generate(obs_placeholder, **fgsm_params)
        adv_obs = attack_op.eval({obs_placeholder: obs[None, :]})

        action = model(adv_obs)[0]
        obs, rew, done, _ = env.step(action)
        episode_rew += rew[0] if isinstance(env, VecEnv) else rew
        env.render()
        done = done.any() if isinstance(done, np.ndarray) else done
        if done:
            print('episode_rew={}'.format(episode_rew))
            episode_rew = 0
            # episode_rew = 0
            obs = env.reset()
    env.close()
    return episode_rew


def main(args):

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

    model, env, q_placeholder, obs_placeholder = _load(args)
    total_reward = eval_model(model, env, q_placeholder, obs_placeholder,
                              eval_steps=args.eval_steps,
                              attack_method=args.attack,
                              eps=args.eps)
    print("total_reward: %.4f" % total_reward)
    # model_path = os.path.join(track.trial_dir(), 'model.ckpt')
    # track.metric(trial_id=args.trial_id, reward=episode_reward,
                 # model_path=model_path)


def _add_args(parser):
    # note: the baselines ddpg implementation is actually buggy, even 
    # out-of-the-box
    parser.add_argument('--alg', default='deepq', help='agent to train',
                        choices=['deepq', 'trpo_mpi', 'ppo2', 'a2c'])
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
    parser.add_argument('--num_trials', default=10, type=int,
                        help='how many times to repeat the experiment')
    parser.add_argument('--load_path', default='', required=True,
                        help='Location of model with correct policy')
    # parser.add_argument('--train', action='store_const',
    #    # help='if true, just trains the plain model from scratch')


if __name__ == '__main__':
    skeletor.supply_args(_add_args)
    skeletor.execute(main)
