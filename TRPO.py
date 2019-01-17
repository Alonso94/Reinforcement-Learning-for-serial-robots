import tensorflow as tf
print(tf.__version__)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
print(keras.__version__)

import gym
import gym_crumb
import numpy as np
from RLagent import RLagent
from utils import *


class TRPO(RLagent):


    def update_policy(self, session, t):
        self.session = session

        states = t['states']
        actions = t['actions']
        advantages = t['advantages']

        feed_dict = {self.input_placeholder: states,
                     self.action_placeholder: actions,
                     self.advantages_placeholder: advantages,
                     self.beta_placeholder: self.beta,
                     self.ksi_placeholder: self.ksi}

        prev_mu, prev_sigma = session.run([self.mu, self.sigma], feed_dict)
        feed_dict[self.previous_mu_placeholder] = np.reshape(prev_mu, (-1, self.env_action_number))
        feed_dict[self.previous_sigma_placeholder] = np.reshape(prev_sigma, (-1, self.env_action_number))

        ##################################
        ### optimization graph for TRPO
        ##################################
        neg_policy_loss = self.policy_loss_function()
        flat_grad_surr = flat_grad1(neg_policy_loss, self._policy_model_params)
        # intermediate grad in conjugate gradient
        cg_inter_vec = tf.placeholder(shape=(None,), dtype=tf.float32)
        # slice flat_tangent into chunks for each weight
        weight_shapes = [session.run(var).shape for var in self._policy_model_params]
        tangents = slice_vector(cg_inter_vec, weight_shapes)

        # kl divergence where the firsts are fixed
        Nf = tf.cast(tf.shape(self.input_placeholder)[0], dtype=tf.float32)
        kl_firstfixed = gauss_selfKL_firstfixed(self.mu, self.sigma) / Nf

        # compute fisher transformation matrix
        grads = tf.gradients(kl_firstfixed, self._policy_model_params)
        gradient_vector_product = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        _fisher_vector_product = flat_grad1(gradient_vector_product, self._policy_model_params)

        # network weights -> vector
        flat_weights = tf.concat([tf.reshape(var, [-1]) for var in self._policy_model_params], axis=0)

        # vector -> network weights
        flat_weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
        assigns = slice_vector(flat_weights_ph, weight_shapes)
        load_flat_weights = [w.assign(ph) for (w, ph) in zip(self._policy_model_params, assigns)]

        old_weights = session.run(flat_weights)
        losses = [neg_policy_loss, self.kl_divergence, self.entropy]

        def fisher_vector_product(p):
            feed_dict[cg_inter_vec] = p
            return session.run(_fisher_vector_product, feed_dict) + self.cg_damping * p

        flat_grad = session.run(flat_grad_surr, feed_dict)
        stepdir = conjugate_gradient(fisher_vector_product, -flat_grad)
        shs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))
        lm = np.sqrt(shs / self.max_kl)
        fullstep = stepdir / lm

        def losses_f(flat_weights):
            feed_dict[flat_weights_ph] = flat_weights
            session.run(load_flat_weights, feed_dict)
            return session.run(losses[0], feed_dict)

        expected_improve_rate = -flat_grad.dot(stepdir)
        new_weights = linear_search(losses_f, old_weights, fullstep, expected_improve_rate)
        feed_dict[flat_weights_ph] = new_weights
        session.run(load_flat_weights, feed_dict)

        neg_policy_loss, kl_divergence, entropy = session.run(losses, feed_dict=feed_dict)
        kl_divergence = np.mean(kl_divergence)

        self.auditor.update({'policy_loss': float("%.5f" % -neg_policy_loss),
                             'kl_divergence': float("%.5f" % kl_divergence),
                             'beta': self.beta,
                             'entropy': float("%.5f" % entropy)})