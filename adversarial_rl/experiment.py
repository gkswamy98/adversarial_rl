""" This contains the code to attack an arbitrary trained agent on a gym env"""
import track
from cleverhans.attacks import FastGradientMethod
from dqn import learn as dqn_learner
from trpo import learn as trpo_learner

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.run import build_env, get_learn_function_defaults,\
                          get_default_network, get_env_type

import os
import numpy as np

_LEARNERS = {
    'dqn': dqn_learner,
    'trpo': trpo_learner
}

_ATTACKS = {
    'fgsm': FastGradientMethod,
}


def _train(args):
    args_dict = vars(args)
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    # create the learner
    learn = _LEARNERS[args.alg]
    attack = _ATTACKS[args.attack_method]  # can't init, it needs the model

    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, os.join(track.trial_dir(), "videos"),
                               record_video_trigger=lambda x: x
                               % args.save_video_interval == 0,
                               video_length=args.save_video_length)

    model = learn(
        attack=attack,
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **args_dict
    )
    return model, env


def eval_model(model, env, eval_steps=10):
    obs = env.reset()
    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))

    episode_rew = 0
    for _ in range(eval_steps):
        if state is not None:
            actions, _, state, _ = model.step(obs, S=state, M=dones)
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
    # track will already handle logging structures
    model, env = _train(args)
    episode_reward = eval_model(model, env, eval_steps=args.eval_steps)
    model_path = os.path.join(track.trial_dir(), 'model.ckpt')
    track.metric(trial_id=args.trial_id, reward=episode_reward,
                 model_path=model_path)
    # TODO dump model here
    trial_dir = os.path.join(args.logdir, 'trial_%d' % args.trial_id)
