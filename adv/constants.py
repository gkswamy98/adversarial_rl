""" constants for all the programs """
from baselines.deepq import deepq
from baselines.trpo_mpi import trpo_mpi
from baselines.ppo2 import ppo2
from baselines.a2c import a2c
from cleverhans.attacks import FastGradientMethod
from adv.attacks import RandomAttack

ATTACKS = {
    'fgsm': FastGradientMethod,
    'random': RandomAttack
}

ALG_LEARN_FNS = {
    'deepq': deepq.learn,
    'trpo_mpi': trpo_mpi.learn,
    'ppo2': ppo2.learn,
    'a2c': a2c.learn
}

VALID_ALGS = list(ALG_LEARN_FNS.keys())

POLICY_GRAD_ALGS = [
    'trpo_mpi',
    'ppo2',
    'a2c'
]
