import skeletor
import matplotlib.pyplot as plt
import numpy as np

import functools
import itertools
import operator
import argparse



keys = ('alg', 'attack', 'eps')
EPS_RANGE = list(np.linspace(0.0, 1.0, steps=10))


def _filtered_df(df, conf):
    attr, val = conf
    return df[functools.reduce(operator.or, df[attr] == val)]


def plot_single_trial(df, label, fig=None):
    # make some pretty plots here


def plot_all_trials(df, configs):

    figure = None   ###
    for conf in configs:
        subfig = None ###
        label = None ###
        plot_single_trial(_filtered_df(df, conf), label, subfig=subfig)


def eval_all_trials(proj, args=None):
    df = skeletor.proc.df_from_proj(proj)
    if args is None:  # this happens if we run right after skeletor finishes
        alg_names = list(experiment._LEARNERS.keys())
        attack_names = list(experiment._ATTACKS.keys())
        keys = [alg_names, attack_names, EPS_RANGE]
        # get all possible configurations
        configs = itertools.product(*keys)
        plot_all_trials(df, configs)
    else:
        alg = args.alg
        attack = args.attack
        eps = args.eps
        df = _filtered_df(df, (alg, attack, eps))
        label = None ###
        plot_single_trial(df, label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='analyze experiment results...')
    parser.add_argument('--logdir', default='./logs')
    parser.add_argument('--alg', default='dqn')
    parser.add_argument('--attack', default='fgsm')
    parser.add_argument('--eps', default=.1)
    parser.add_argument('experimentname', default='dqn')
    args = parser.parse_args()

    proj = skeletor.proc.proj(args.experimentname, args.logdir)
    eval_all_trials(proj, args=args)

