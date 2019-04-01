import skeletor
from .experiment import main
from .analysis import eval_all_trials

import os


def _logdir_name(args):
    base = args.logdir
    args_dict = vars(args)
    attrs = ['alg', 'env', 'attack']
    subdir = sum('_%s=%s' % (attr, args_dict[attr]) for attr in attrs)[1:]
    return os.path.join(base, subdir)


def add_args(parser):
    # add the rest....
    parser.add_argument('--alg', default='deepq', help='agent to train',
                        choices=['deepq', 'mpi_trpo'])
    parser.add_argument('--env', default='PongNoFrameskip-v4',
                        help='Gym environment name to train on')
    parser.add_argument('--attack', default='fgsm',
                        choices=['fgsm'],
                        help='attack method to run')
    parser.add_argument('--attack-norm', default='l1',
                        choices=['l1', 'l2', 'l0'],
                        help="norm we use to constrain perturbation size")
    parser.add_argument('--eps', default=.1, type=float,
                        help='perturbation magnitude')
    parser.add_argument('--num-trials', default=10, type=int,
                        help='how many times to repeat the experiment')
    parser.add_argument('--load_path', default='', required=True,
                        help='Location of model with correct policy')
    parser.add_argument('--train', action='store_const',
                        help='if true, just trains the plain model from scratch')


if __name__ == '__main__':
    skeletor.supply_args(add_args)
    skeletor.supply_postprocess(eval_all_trials)
    skeletor.execute(main)
