import tensorflow as tf
print(tf.__version__)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
print(keras.__version__)

import gym
import mujoco_py
import gym_crumb
import numpy as np
from RLagent import RLagent

class PPO(RLagent):

    def update_policy(self, session, t):
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

        neg_policy_loss, kl_divergence, entropy = 0.0, 0.0, 0.0
        for _ in range(self.n_policy_epochs):
            session.run(self.optimize_policy, feed_dict=feed_dict)
            neg_policy_loss, kl_divergence, entropy = session.run([self.neg_policy_loss, self.kl_divergence, self.entropy], feed_dict=feed_dict)
            kl_divergence = np.mean(kl_divergence)
            if kl_divergence > 4 * self.kl_target:
                break

        self.auditor.update({'policy_loss': float("%.5f" % -neg_policy_loss),
                             'kl_divergence': float("%.5f" % kl_divergence),
                             'beta': self.beta,
                             'entropy': float("%.5f" % entropy)})

        if kl_divergence < self.kl_target / 1.5:
            self.beta /= 2
        elif kl_divergence > self.kl_target * 1.5:
            self.beta *= 2
        self.beta = np.clip(self.beta, self.beta_min, self.beta_max)