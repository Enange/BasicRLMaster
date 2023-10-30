import torch as T
import torch.nn as nn
from collections import deque
import numpy as np
import gymnasium as gym
import random

# LOOP 3 chiamate a funzioni non esistenti

# Create the Mode
class Network(nn.Module):
    def __init__(self, input_shape, output_size, hidden=64):
        # Input -> 64 -> 64 -> output
        super(Network, self).__init__()
        self.input_layer = nn.Linear(in_features=input_shape, out_features=hidden)
        self.hidden = nn.Linear(in_features=hidden, out_features=hidden)
        self.output_layer = nn.Linear(in_features=hidden, out_features=output_size)

        self.relu = nn.ReLU()


    def forward(self, x):
        return self.output_layer(self.relu(self.hidden(self.input_layer(x))))


class DQN:
    def __init__(self, env, verbose):
        # Prende enviroment da Gym, Verbose mostra l'avanzamento
        self.env = env
        self.verbose = verbose
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.input_shape = self.env.observation_space.shape  # Input possibili
        self.action_space = env.action_space.n  # Output possibili
        self.actor = Network(self.input_shape, self.action_space).to(self.device) #Ritorna un modello neurale dati input e output

        self.actor_target = Network(self.input_shape, self.action_space).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())      #Copia dei Parametri

        self.optimizer = T.optim.Adam(self.actor.parameters())    # OTTIMIZZA
        self.gamma = 0.95   # Ammortamento Premi
        self.memory_size = 2000 #Dimensione Memoria
        self.batch_size = 32    #Numeri campioni propagati nella rete
        self.exploration_rate = 1.0 #Tasso iniziale di Exploration
        self.exploration_decay = 0.995 # Fattore di decadimento
        self.tau = 0.005

        self.run_id = np.random.randint(0, 1000) # ritorna un numero intero tra una distribuzione discreta per salvare i dati /data/...
        self.render = False

    def loop(self, num_episodes=1000):
        reward_list = []
        ep_reward_mean = deque(maxlen=100)  # usato per i dati
        replay_buffer = deque(maxlen=self.memory_size)  # lista per salvare [state, action, reward, new_state, done]

        # ciclo per tutti gli episodi (in example)
        for episode in range(num_episodes):
            state, info = self.env.reset(seed=123, options={})
            ep_reward = 0  # REset reward ad ogni tentativo

            while True:
                if self.render: self.env.render()
                action = self.get_action(state)  # Ottengo l'azione da fare
                new_state, reward, terminated, truncated, _ = self.env.step(
                    action)  # Vedo il risultato dell'azione nell' enviroment
                ep_reward += reward  # Somma dei rewards dell'episodio
                done = terminated or truncated  # Mi rendo conto se l'azione è fallita o terminata -> rifaccio partire il mio enviroment

                replay_buffer.append([state, action, reward, new_state,
                                      done])  # Aggiorno replay buffer( record dei [state, action, reward, new_state, done])

                # Episodio finito
                if done: break
                state = new_state  # Aggiorno lo stato in cui sono attualmente

                self.update_networks(replay_buffer)
                self._update_target(self.actor.variables, self.actor_target.variables, tau=self.tau)

            self.exploration_rate = self.exploration_rate * self.exploration_decay if self.exploration_rate > 0.05 else 0.05  # Aggiorno exploraion rate
            ep_reward_mean.append(ep_reward)  # Ridondante
            reward_list.append(ep_reward)  # salvo i miei punteggi
            # Salvo i miei dati
            if self.verbose > 0: print(
                f"Episode: {episode:7.0f}, reward: {ep_reward:8.2f}, mean_last_100: {np.mean(ep_reward_mean):8.2f}, exploration: {self.exploration_rate:0.2f}")
            if self.verbose > 1: np.savetxt(f"data/reward_DQN_{self.run_id}.txt", reward_list)
