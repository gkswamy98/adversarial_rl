""" custom attacks we need to run (vs. just FGSM) for ablation studies """
import tensorflow as tf


class RandomAttack:
    """
    Generates a random vector with l_inf norm of the given magnitude (eps)
    """
    def __init__(self, model, **kwargs):
        pass

    def generate(self, obs_placeholder, eps=0.1, **kwargs):
        x = tf.random_normal([1, int(obs_placeholder.shape[1])])
        return obs_placeholder + eps * x / tf.norm(x, ord=2)
