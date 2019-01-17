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
import scipy.signal
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Input, Conv2D,Dropout,Flatten,MaxPooling2D


from utils import *

class RLagent(object):

    def __init__(self, env,from_images=False,from_images_mujoco=False):
        ### Environment parameters
        tf.reset_default_graph()
        self.env = env
        self.env.reset()
        self.from_images_mujoco=from_images_mujoco
        if from_images_mujoco:
            state = self.env.render(mode='rgb_array')
            self.env_state_shape = state.shape
        else:
            self.env_state_shape = self.env.observation_space.shape
        self.env_action_number = self.env.action_space.shape[0]

        ### Hyperparameters
        self.policy_learning_rate = 0.001
        self.value_learning_rate = 0.001
        self.n_policy_epochs = 20
        self.n_value_epochs = 15
        self.value_batch_size = 512
        self.kl_target = 0.003
        self.cg_damping = 0.1
        self.max_kl = 0.01
        self.beta = 1
        self.beta_max = 20
        self.beta_min = 1 / 20
        self.ksi = 10
        self.gamma = 0.999
        self.lmbda = 0.975
        self.horizon = 64
        self.activation = 'tanh'
        self.policy_type = 'MLP'  # or 'RBF'

        ### For train
        self.max_episode_count = 1000
        self.episode_count = 0
        self.auditor = {}
        self.running_stats = RunningStats(self.env_state_shape)
        self.reward_scale = 0.0025

        ### Placeholders
        if from_images_mujoco:
            obs_in_in=[self.env_state_shape[0],self.env_state_shape[1],1]
        if from_images:
            obs_in_in=[self.env_state_shape[0],self.env_state_shape[1],self.env_state_shape[2]]
        else:
            obs_in_in=[len(self.env.observation_space.high)]
        self.input_placeholder = tf.placeholder(shape=[None, obs_in_in[:]], dtype=tf.float32,
                                                name="input")
        self.reward = tf.placeholder(shape=[None, 1], dtype='float32', name='reward')
        self.action_placeholder = tf.placeholder(shape=[None, self.env_action_number], dtype='float32', name='actions')
        self.advantages_placeholder = tf.placeholder(shape=[None,1], dtype='float32', name='GAE_advantage')
        self.previous_mu_placeholder = tf.placeholder(shape=[None, self.env_action_number], dtype='float32',
                                                      name='previous_iteration_mu')
        self.previous_sigma_placeholder = tf.placeholder(shape=[None, self.env_action_number], dtype='float32',
                                                         name='previous_iteration_sigma')
        self.beta_placeholder = tf.placeholder(shape=[], dtype='float32', name='beta_2nd_loss')
        self.ksi_placeholder = tf.placeholder(shape=[], dtype='float32', name='eta_3rd_loss')

        self.build_networks(from_images or from_images_mujoco)

        ### loss function and optimization
        self.neg_policy_loss = self.policy_loss_function()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.policy_learning_rate)
        self.optimize_policy = self.optimizer.minimize(self.neg_policy_loss,
                                                       var_list=self._policy_model_params.append(self.sigma))

        self.sampled_action = tf.squeeze(self.mu + self.sigma * tf.random_normal(shape=tf.shape(self.mu)))

    def build_networks(self,from_images):

        ### Action mean network
        mu_model_input = Input(tensor=self.input_placeholder)
        if from_images:
            h1 = Conv2D(32, kernel_size=(3, 3), padding="same", activation="elu")(mu_model_input)
            pool1 = MaxPooling2D(pool_size=(2, 2))(h1)
            h2 = Conv2D(64, kernel_size=(3, 3), padding="same", activation="elu")(pool1)
            features = Flatten()(h2)
            mu_model = Dense(units=128, activation=self.activation, kernel_initializer=RandomNormal(0, 0.1))(features)
        else:
            mu_model = Dense(units=128, activation=self.activation, kernel_initializer=RandomNormal(0, 0.1))(mu_model_input)
        mu_model = Dense(units=128, activation=self.activation, kernel_initializer=RandomNormal(0, 0.1))(mu_model)
        policy_mean = Dense(units=self.env_action_number, activation=self.activation,
                            kernel_initializer=RandomNormal())(mu_model)
        self.policy_mu_model = Model(inputs=[mu_model_input], outputs=[policy_mean])

        self._policy_model_params = self.policy_mu_model.trainable_weights

        ### State value network
        value_model_input = Input(batch_shape=(None, self.env_state_shape[0]))
        value_model = Dense(units=128, activation=self.activation, kernel_regularizer=l2(0.01))(value_model_input)
        value_model = Dense(units=128, activation=self.activation, kernel_regularizer=l2(0.01))(value_model)
        value = Dense(units=1, activation=None, kernel_initializer=RandomNormal(0, 0.1), kernel_regularizer=l2(0.01))(
            value_model)
        self.value_model = Model(inputs=[value_model_input], outputs=[value])

        self._value_model_params = self.value_model.trainable_weights

        self.adam_optimizer = Adam(self.value_learning_rate)
        self.value_model.compile(loss='mean_squared_error', optimizer=self.adam_optimizer)

        ### model outputs
        self.mu = self.policy_mu_model(self.input_placeholder)
        self.value = self.value_model(self.input_placeholder)
        self.sigma = tf.get_variable('sigma', (1, self.env_action_number), tf.float32, tf.constant_initializer(0.6))

        self.policy_mu_model.summary()
        self.value_model.summary()
        return self

    def policy_loss_function(self):
        normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
        prev_normal_dist = tf.contrib.distributions.Normal(self.previous_mu_placeholder,
                                                           self.previous_sigma_placeholder)
        self.logp = (normal_dist.log_prob(self.action_placeholder))
        self.prev_logp = (prev_normal_dist.log_prob(self.action_placeholder))

        self.kl_divergence = tf.contrib.distributions.kl_divergence(normal_dist, prev_normal_dist)
        self.entropy = tf.reduce_mean(tf.reduce_sum(normal_dist.entropy(), axis=1))

        self.ratio=tf.exp(self.logp - self.prev_logp)

        ### adaptive kl penality coefficient
        negloss = -tf.reduce_mean(self.advantages_placeholder * self.ratio)
        negloss += tf.reduce_mean(self.beta_placeholder * self.kl_divergence)
        negloss += tf.reduce_mean(
            self.ksi_placeholder * tf.square(tf.maximum(0.0, self.kl_divergence - 2 * self.kl_target)))
        return negloss

    def sample_action(self, session, input):
        return session.run(self.sampled_action,
                           feed_dict={self.input_placeholder: np.reshape(input, (-1, self.env_state_shape[0]))})

    def update_policy(self, session, t):
        pass

    def update_value(self, t):
        states = t['states']
        target_values = t['disc_rewards']
        self.value_model.fit(x=states, y=target_values, epochs=self.n_value_epochs, batch_size=self.value_batch_size,
                             verbose=0)
        value_loss = self.value_model.evaluate(x=states, y=target_values, verbose=0)
        self.auditor.update({'value_loss': float("%.7f" % value_loss)})
        return self

    #####################
    # Train
    #####################

    def gen_trajectory(self, session):
        state = self.env.reset()
        actions, rewards, states, norm_states = [], [], [], []
        terminal = False
        while terminal is False:
            states.append(state)
            state_normalized = (state - self.running_stats.mean()) / self.running_stats.standard_deviation()
            norm_states.append(state_normalized)

            action = self.sample_action(session, state_normalized)
            new_obs, reward, terminal, info = self.env.step(action)
            if self.from_images_mujoco:
                new_state = self.env.render(mode='rgb_array')
            else:
                new_state=new_obs
            self.env.render()
            actions.append(action)
            rewards.append(reward * self.reward_scale)

            state = new_state

        return actions, rewards, states, norm_states

    def discount(self, x, gamma):
        return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

    def rollout(self, session, horizon):
        raw_t = {'states': [], 'actions': [], 'rewards': [], 'disc_rewards': [], 'values': [], 'advantages': []}
        raw_states = []
        for episode in range(horizon):
            actions, rewards, states, norm_states = self.gen_trajectory(session)
            raw_t['states'].append(norm_states)
            raw_t['actions'].append(actions)
            raw_t['rewards'].append(rewards)
            raw_t['disc_rewards'].append(self.discount(rewards, gamma=self.gamma))
            raw_states += states
            self.episode_count += 1

        self.running_stats.multiple_push(raw_states)

        self.auditor.update({'episode_number': self.episode_count,
                             'per_episode_mean': int(
                                 np.sum(np.concatenate(raw_t['rewards'])) / (horizon * self.reward_scale))})

        return raw_t

    def initialize_params(self, session, n_episode):
        self.rollout(session, n_episode)

    def process_trajectories(self, session, t):
        for i in range(self.horizon):
            feed_dict = {self.input_placeholder: t['states'][i]}
            values = session.run(self.value, feed_dict=feed_dict)
            t['values'].append(values)

            temporal_differences = t['rewards'][i] + np.append(self.gamma * values[1:], 0.0) - list(map(float, values))
            gae = self.discount(temporal_differences, self.lmbda * self.gamma)
            t['advantages'].append(gae)

        t['states'] = np.concatenate(t['states'])
        t['actions'] = np.concatenate(t['actions'])
        t['rewards'] = np.concatenate(t['rewards'])
        t['disc_rewards'] = np.concatenate(t['disc_rewards'])
        t['values'] = np.concatenate(t['values'])

        concatenated_gae = np.concatenate(t['advantages'])
        normalized_gae = (concatenated_gae - concatenated_gae.mean()) / (concatenated_gae.std() + 1e-6)

        t['advantages'] = normalized_gae

        t['actions'] = np.reshape(t['actions'], (-1, self.env_action_number))
        for i in ['rewards', 'disc_rewards', 'values', 'advantages']:
            t[i] = np.reshape(t[i], (-1, 1))
        return t

    def train(self, session):
        with session.as_default(), session.graph.as_default():
            self.initialize_params(session=session, n_episode=3)

            raw_t = self.rollout(session, self.horizon)
            t_processed = self.process_trajectories(session, raw_t)
            t_processed_prev = t_processed
            print("begin training ...")

            while self.episode_count < self.max_episode_count:

                raw_t = self.rollout(session, self.horizon)
                t_processed = self.process_trajectories(session, raw_t)

                self.update_policy(session, t_processed)
                self.update_value(t_processed_prev)

                print(self.auditor)

                t_processed_prev = t_processed
