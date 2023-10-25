import torch
import torch.nn as nn
from collections import deque
import numpy as np
import gymnasium as gym
import random


class DQN(nn.Module):
    def __init__(self, env, verbose):
        super().__init__()
        self.relu = None
        self.layer_1 = None
        self.layer_2 = None
        self.layer_3 = None
        self.env = env
        self.verbose = verbose

        self.input_shape = self.env.observation_space.shape
        self.action_space = env.action_space.neììr
        self.actor = self.get_actor_model(self.input_shape, self.action_space)
        self.actor_target = self.get_actor_model(self.input_shape, self.action_space)
        self.actor_target.set_weights(self.actor.get_weights())

        self.optimizer = torch.optim.Adam()
        self.gamma = 0.95
        self.memory_size = 2000
        self.batch_size = 32
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.tau = 0.005

        self.run_id = np.random.randint(0, 1000)
        self.render = False



    def loop(self, num_episodes=1000):
        reward_list = []
        ep_reward_mean = deque(maxlen=100)
        replay_buffer = deque(maxlen=self.memory_size)

        for episode in range(num_episodes):
            state, info = self.env.reset(seed=123, options={})
            ep_reward = 0

            while True:
                if self.render: self.env.render()
                action = self.get_action(state)
                new_state, reward, terminated, truncated, _ = self.env.step(action)
                ep_reward += reward
                done = terminated or truncated

                replay_buffer.append([state, action, reward, new_state, done])

                if done: break
                state = new_state

                self.update_networks(replay_buffer)
                self._update_target(self.actor.variables, self.actor_target.variables, tau=self.tau)

            self.exploration_rate = self.exploration_rate * self.exploration_decay if self.exploration_rate > 0.05 else 0.05
            ep_reward_mean.append(ep_reward)
            reward_list.append(ep_reward)
            if self.verbose > 0: print(
                f"Episode: {episode:7.0f}, reward: {ep_reward:8.2f}, mean_last_100: {np.mean(ep_reward_mean):8.2f}, "
                f"exploration: {self.exploration_rate:0.2f}")
            if self.verbose > 1: np.savetxt(f"data/reward_DQN_{self.run_id}.txt", reward_list)

    def _update_target(self, weights, target_weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def get_action(self, state):
        if np.random.random() < self.exploration_rate:
            return np.random.choice(self.action_space)
        return np.argmax(self.actor(state.reshape((1, -1))))

    def update_networks(self, replay_buffer):
        samples = np.array(random.sample(replay_buffer, min(len(replay_buffer), self.batch_size)), dtype=object)
        with tf.GradientTape() as tape:
            objective_function = self.actor_objective_function_double(samples)  # Compute loss with custom loss function
            grads = tape.gradient(objective_function,
                                  self.actor.trainable_variables)  # Compute gradients actor for network
            self.optimizer.apply_gradients(
                zip(grads, self.actor.trainable_variables))  # Apply gradients to update network weights

    def actor_objective_function_double(self, replay_buffer):
        state = np.vstack(replay_buffer[:, 0])
        action = replay_buffer[:, 1]
        reward = np.vstack(replay_buffer[:, 2])
        new_state = np.vstack(replay_buffer[:, 3])
        done = np.vstack(replay_buffer[:, 4])

        next_state_action = np.argmax(self.actor(new_state), axis=1)
        target_mask = self.actor_target(new_state) * tf.one_hot(next_state_action, self.action_space)
        target_mask = tf.reduce_sum(target_mask, axis=1, keepdims=True)

        target_value = reward + (1 - done.astype(int)) * self.gamma * target_mask
        mask = self.actor(state) * tf.one_hot(action, self.action_space)
        prediction_value = tf.reduce_sum(mask, axis=1, keepdims=True)

        mse = tf.math.square(prediction_value - target_value)
        return tf.math.reduce_mean(mse)

    def get_actor_model(self, input_shape, output_size):
        self.layer_1 = nn.Linear(in_features=self.input_shape, out_features=64)
        self.layer_2 = nn.Linear(in_features=64, out_features=64)
        self.layer_3 = nn.Linear(in_features=64, out_features=self.action_space)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))



    ##########################
    #### VANILLA METHODS #####
    ##########################
    def actor_objective_function_fixed_target(self, replay_buffer):
        state = np.vstack(replay_buffer[:, 0])
        action = replay_buffer[:, 1]
        reward = np.vstack(replay_buffer[:, 2])
        new_state = np.vstack(replay_buffer[:, 3])
        done = np.vstack(replay_buffer[:, 4])

        target_value = reward + (1 - done.astype(int)) * self.gamma * np.amax(self.actor_target(new_state), axis=1,
                                                                              keepdims=True)
        mask = self.actor(state) * tf.one_hot(action, self.action_space)
        prediction_value = tf.reduce_sum(mask, axis=1, keepdims=True)

        mse = tf.math.square(prediction_value - target_value)
        return tf.math.reduce_mean(mse)

    def actor_objective_function_std(self, replay_buffer):
        state = np.vstack(replay_buffer[:, 0])
        action = replay_buffer[:, 1]
        reward = np.vstack(replay_buffer[:, 2])
        new_state = np.vstack(replay_buffer[:, 3])
        done = np.vstack(replay_buffer[:, 4])

        target_value = reward + (1 - done.astype(int)) * self.gamma * np.amax(self.actor(new_state), axis=1,
                                                                              keepdims=True)
        mask = self.actor(state) * tf.one_hot(action, self.action_space)
        prediction_value = tf.reduce_sum(mask, axis=1, keepdims=True)

        mse = tf.math.square(prediction_value - target_value)
        return tf.math.reduce_mean(mse)
