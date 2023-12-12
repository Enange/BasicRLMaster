import torch
import torch as T
import torch.nn as nn
from collections import deque
import numpy as np
import gym
import random

'''
COSE DA SISTEMARE
s 
*device
*Neural network Continuos
'''
    ##
    # DISCRETE
    ##

class Network_disc(nn.Module):
    def __init__(self, input_shape, output_size, hiddenNodes=32):
        # Input -> 64 -> 64 -> output
        super(Network_disc, self).__init__()
        self.input_layer = nn.Linear(in_features=input_shape, out_features=hiddenNodes)
        self.hidden = nn.Linear(in_features=hiddenNodes, out_features=hiddenNodes)
        self.hidden2 = nn.Linear(in_features=hiddenNodes, out_features=hiddenNodes)
        self.output_layer = nn.Linear(in_features=hiddenNodes, out_features=output_size)  # np.array(output_size).prod())

    def forward(self, x):
        # x = nn.functional.relu(self.input_layer(x))
        x = self.input_layer(x)
        x = nn.functional.relu(self.hidden(x))
        x = nn.functional.relu(self.hidden2(x))

        return nn.functional.softmax(self.output_layer(x))
    ##
    # CONTINUOS
    ##

class REINFORCE_PT:
    def __init__(self, env, discrete, verbose):
        self.env = env
        self.discrete = discrete
        self.verbose = verbose

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.input_shape = self.env.observation_space.shape[0]
        if self.discrete:
            self.action_space = env.action_space.n
        else:
            self.action_space = env.action_space.shape[0]

        if self.discrete:
            self.actor = Network_disc(self.input_shape, self.action_space).to(self.device)
            self.get_action = self.get_action_disc
            self.actor_objective_function = self.actor_objective_function_disc
        # else:
        #     self.actor = self.get_actor_model_cont(self.input_shape, self.action_space,
        #                                            [env.action_space.low, env.action_space.high])
        #     #self.get_action = self.get_action_cont
        #     self.actor_objective_function = self.actor_objective_function_cont
        #

        self.optimizer = T.optim.Adam(self.actor.parameters())
        self.gamma = 0.99
        self.sigma = 1.0
        self.exploration_decay = 1

        self.run_id = np.random.randint(0, 1000)
        self.render = False

    def loop(self, num_episodes=1000):
        reward_list = []
        ep_reward_mean = deque(maxlen=100)
        memory_buffer = deque()

        for episode in range(num_episodes):
            state, info = self.env.reset(seed=123, options={})
            ep_reward = 0

            while True:
                if self.render: self.env.render()
                action = self.get_action(state)
                new_state, reward, terminated, truncated, _ = self.env.step(action)
                ep_reward += reward
                done = terminated or truncated

                memory_buffer.append([state, reward, action])
                if done: break
                state = new_state

            self.update_networks(np.array(memory_buffer, dtype=object))
            memory_buffer.clear()
            self.sigma = self.sigma * self.exploration_decay if self.sigma > 0.05 else 0.05

            ep_reward_mean.append(ep_reward)
            reward_list.append(ep_reward)
            if self.verbose > 0 and not self.discrete: print(
                f"Episode: {episode:7.0f}, reward: {ep_reward:8.2f}, mean_last_100: {np.mean(ep_reward_mean):8.2f}, sigma: {self.sigma:0.2f}")
            if self.verbose > 0 and self.discrete: print(
                f"Episode: {episode:7.0f}, reward: {ep_reward:8.2f}, mean_last_100: {np.mean(ep_reward_mean):8.2f}")
            if self.verbose > 1: np.savetxt(f"data/reward_REINFORCEPT_{self.run_id}.txt", reward_list)

    def update_networks(self, memory_buffer):
        memory_buffer[:, 1] = self.discount_reward(memory_buffer[:, 1])  # Discount the rewards in a MC way

        objective_function = self.actor_objective_function(memory_buffer)# Compute loss with custom loss function

        self.optimizer.zero_grad()
        objective_function.backward()
        self.optimizer.step()

        # with tf.GradientTape() as tape:
        #     objective_function = self.actor_objective_function(memory_buffer)  # Compute loss with custom loss function
        #     grads = tape.gradient(objective_function,
        #                           self.actor.trainable_variables)  # Compute gradients actor for network
        #     self.optimizer.apply_gradients(
        #         zip(grads, self.actor.trainable_variables))  # Apply gradients to update network weights


    def discount_reward(self, rewards):
        sum_reward = 0
        discounted_rewards = []

        for r in rewards[::-1]:
            sum_reward = r + self.gamma * sum_reward
            discounted_rewards.append(sum_reward)
        discounted_rewards.reverse()

        # Normalize
        eps = np.finfo(np.float64).eps.item()  # Smallest number such that 1.0 + eps != 1.0
        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + eps)

        return discounted_rewards

    ## DISCRETE

    def get_action_disc(self, state):
        state = state.reshape(1,-1)


        softmax_out = self.actor(T.tensor(state)) #PROBLEMA SUL RESHAPE -> 122 riga

        selected_action = np.random.choice(self.action_space, p=softmax_out.detach().numpy()[0])
        return selected_action


    def actor_objective_function_disc(self, memory_buffer):
        # Extract values from buffer
        state = T.from_numpy(np.vstack(memory_buffer[:, 0])).float().to(self.device)
        reward = np.vstack(memory_buffer[:, 1])
        action = T.tensor(list(memory_buffer[:, 2])).to(self.device)

        baseline = np.mean(reward)
        probability = self.actor(state)
        # action_idx = [[counter, val] for counter, val in enumerate(action)]


        action_idx = []
        for val in action:
            action_idx.append([val])

        #action_idx = [val for val in enumerate(action)]
        action_idx = T.tensor(action_idx)
        #probability = tf.expand_dims(tf.gather_nd(probability, action_idx), axis=-1)
        # probability = T.tensor(probability)
        probability = torch.gather(probability, dim=1, index=action_idx)
        #probability = T.unsqueeze(probability, dim=-1)
        #partial_objective = tf.math.log(probability) * (reward - baseline)

        partial_objective = T.log(probability) * T.tensor(reward - baseline)
        return -T.mean(partial_objective)


