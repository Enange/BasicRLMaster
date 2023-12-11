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


class Network_disc(nn.Module):
    def __init__(self, input_shape, output_size, hidden=32):
        # Input -> 64 -> 64 -> output
        super(Network_disc, self).__init__()
        self.input_layer = nn.Linear(in_features=input_shape, out_features=hidden)
        self.hidden = nn.Linear(in_features=hidden, out_features=hidden)
        self.hidden2 = nn.Linear(in_features=hidden, out_features=hidden)
        self.output_layer = nn.Linear(in_features=hidden, out_features=output_size)  # np.array(output_size).prod())

    def forward(self, x):
        # x = nn.functional.relu(self.input_layer(x))
        x = self.input_layer(x)
        x = nn.functional.relu(self.hidden(x))
        x = nn.functional.relu(self.hidden2(x))

        return nn.functional.softmax(self.output_layer(x))

class Network_const(nn.Module):

class REINFORCE_PT:
    def __init__(self, env, discrete, verbose):
        self.env = env
        self.discrete = discrete
        self.verbose = verbose

        # self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.input_shape = self.env.observation_space.shape
        if self.discrete:
            self.action_space = env.action_space.n
        else:
            self.action_space = env.action_space.shape[0]

        if self.discrete:
            self.actor = self.get_actor_model_disc(self.input_shape, self.action_space)
            self.get_action = self.get_action_disc
            self.actor_objective_function = self.actor_objective_function_disc
        else:
            self.actor = self.get_actor_model_cont(self.input_shape, self.action_space,
                                                   [env.action_space.low, env.action_space.high])
            self.get_action = self.get_action_cont
            self.actor_objective_function = self.actor_objective_function_cont

        self.optimizer = keras.optimizers.Adam()
        self.gamma = 0.99
        self.sigma = 1.0
        self.exploration_decay = 1

        self.run_id = np.random.randint(0, 1000)
        self.render = False
