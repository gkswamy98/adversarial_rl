import skeletor
from baselines.run import main as run


def make_args(parser):
    pass

def main(args):
    # just the args back to the gym ones... see experiment.py
    ## TODO do that
    run(args)

if __name__ == '__main__':
    skeletor.supply_args(make_args)
    skeletor.execute(main)

