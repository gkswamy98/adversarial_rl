""" custom attacks we need to run (vs. just FGSM) for ablation studies """
import tensorflow as tf
import tensorflow_probability as tfp


class RandomAttack:
    """
    Generates a random vector with l_p norm of the given magnitude (eps)
    """
    def __init__(self, model, **kwargs):
        pass

    def generate(self, obs_placeholder, eps=0.1, p=2, **kwargs):
    	if p == 2:
        	x = tf.random_normal([1, int(obs_placeholder.shape[1])])
        	return obs_placeholder + eps * x / tf.norm(x, ord=2)
        if p == 1:
        	x = tfp.distributions.Laplace(loc=0, scale=1).sample([1, int(obs_placeholder.shape[1])])
        	return obs_placeholder + eps * x / tf.norm(x, ord=1)
