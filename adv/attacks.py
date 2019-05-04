""" custom attacks we need to run (vs. just FGSM) for ablation studies """
import numpy as np
import tensorflow as tf


class RandomAttack:
    def __init__(self, model, **kwargs):
        pass

    def generate(self, obs_placeholder, eps=0.1, ord=None):
        x = tf.random_normal([1, 4])
        x = tf.sign(x)
        return obs_placeholder + eps * x[None, :]
        # x = np.random.laplace(loc=0., scale=1., size=obs_placeholder.shape)
        # pert = tf.convert_to_tensor(eps * x / np.linalg.norm(x, ord=1))
        # return tf.to_float(obs_placeholder) + tf.to_float(pert)