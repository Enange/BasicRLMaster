import tensorboard.plugins.projector.projector_plugin
import torch
import torch as T
import torch.nn as nn
from collections import deque
import numpy as np
import gym
import random


# FORWARD Must be passed a Tensor

# Create the Mode
class Network(nn.Module):
    def __init__(self, input_shape, output_size, hidden=64):
        # Input -> 64 -> 64 -> output
        super(Network, self).__init__()
        self.input_layer = nn.Linear(in_features=input_shape, out_features=hidden)
        self.hidden = nn.Linear(in_features=hidden, out_features=hidden)
        self.output_layer = nn.Linear(in_features=hidden, out_features=output_size)  # np.array(output_size).prod())

    def forward(self, x):
        x = nn.functional.relu(self.input_layer(x))
        x = nn.functional.relu(self.hidden(x))

        return self.output_layer(x)


class DQN:
    def __init__(self, env, verbose):
        # Prende enviroment da Gym, Verbose mostra l'avanzamento
        self.env = env
        self.verbose = verbose
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.input_shape = self.env.observation_space.shape[0]  # Input possibili
        self.action_space = env.action_space.n  # Output possibili
        self.actor = Network(self.input_shape, self.action_space).to(
            self.device)  # Ritorna un modello neurale dati input e output

        print(self.action_space)


        self.actor_target = Network(self.input_shape, self.action_space).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())  # Copia dei Parametri

        self.optimizer = T.optim.Adam(self.actor.parameters())  # OTTIMIZZA
        self.gamma = 0.95  # Ammortamento Premi
        self.memory_size = 2000  # Dimensione Memoria
        self.batch_size = 32  # Numeri campioni propagati nella rete
        self.exploration_rate = 1.0  # Tasso iniziale di Exploration
        self.exploration_decay = 0.995  # Fattore di decadimento
        self.tau = 0.005

        self.run_id = np.random.randint(0,
                                        1000)  # ritorna un numero intero tra una distribuzione discreta per salvare i dati /data/...
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
                # self._update_target(self.actor.variables, self.actor_target.variables, tau=self.tau)
                self._update_target(self.actor.parameters(), self.actor_target.parameters(), tau=self.tau)

            self.exploration_rate = self.exploration_rate * self.exploration_decay if self.exploration_rate > 0.05 else 0.05  # Aggiorno exploraion rate
            ep_reward_mean.append(ep_reward)  # Ridondante
            reward_list.append(ep_reward)  # salvo i miei punteggi
            # Salvo i miei dati
            if self.verbose > 0: print(
                f"Episode: {episode:7.0f}, reward: {ep_reward:8.2f}, mean_last_100: {np.mean(ep_reward_mean):8.2f}, exploration: {self.exploration_rate:0.2f}")
            if self.verbose > 1: np.savetxt(f"data/reward_DQN_{self.run_id}.txt", reward_list)

    def _update_target(self, weights, target_weights, tau):
        for (a, b) in zip(target_weights, weights):
            # a.assign(b * tau + a * (1 - tau))
            a.data.copy_(b * tau + a * (1 - tau))

    def get_action(self, state):

        # state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # Azione randomica all'inizio sarà sicuramente randomica
        # exploration rate = 1 e 0 <= random <= 1
        # pian piano si abbassa l'exploration rate e non farà più azioni casuali
        if np.random.random() < self.exploration_rate:
            return np.random.choice(self.action_space)  # a Caso dalle scelte

        state = state.reshape(1, -1)

        action_val = self.actor(T.tensor(state))
        # print("\n\n")
        # print(action_val)
        # print(action_val.cpu())
        # print(action_val.cpu().data)
        # print(np.argmax(action_val.cpu().data.numpy()))
        # print(action_val.cpu().data.numpy())
        return np.argmax(action_val.cpu().data.numpy())

        # return np.argmax(self.actor(state.reshape((1, -1))))  # Scelta data dalla rete neurale

    def update_networks(self, replay_buffer):
        samples = np.array(random.sample(replay_buffer, min(len(replay_buffer), self.batch_size)),
                           dtype=object)  # Prendo un Campione per la mia rete neurale
        with T.no_grad():  # file = open()
            objective_function = self.actor_objective_function_double(samples)  # Compute loss with custom loss function

            objective_function.requires_grad = True
            self.optimizer.zero_grad()
            objective_function.backward()
            self.optimizer.step()  # Apply gradients to update network weights

    def actor_objective_function_double(self, replay_buffer):
        state = torch.from_numpy(np.vstack(replay_buffer[:, 0])).float().to(self.device)  # Prende dal RB lo stato
        # listT = list()
        # for obj in replay_buffer[:,1]:
        #     listT.append(obj)
        # #print(listT)
        ## MAGHEGGIO STRANISSIMO DA SISTEMARE (trasformato il replay buffer in lista,
        ## perchè
        # print(list(replay_buffer[:,1]));
        action = T.tensor(list(replay_buffer[:, 1])).to(self.device)  # Prende dal RB l'azione
        reward = torch.from_numpy(np.vstack(replay_buffer[:, 2])).to(self.device)  # Prende dal RB il reward
        new_state = torch.from_numpy(np.vstack(replay_buffer[:, 3])).float().to(
            self.device)  # Prende dal RB il nuovo stato
        done = torch.from_numpy(np.vstack(replay_buffer[:, 4]).astype(np.int8)).float().to(
            self.device)  # Prende dal RB il done

        next_state_action = np.argmax(self.actor.forward(new_state),
                                      axis=1)  # Calcolo le prossime azioni e ritorna 0 e 1

        target_mask = self.actor_target.forward(new_state) * nn.functional.one_hot(T.tensor(next_state_action),
                                                                                   self.action_space)
        target_mask = T.sum(target_mask, dim=1, keepdim=True)  # Sommando ogni riga, in un valore
        target_value = reward + (1 - done) * self.gamma * target_mask
        mask = self.actor(T.tensor(state)) * nn.functional.one_hot(action, self.action_space)
        prediction_value = T.sum(mask, dim=1, keepdim=True)

        mse = T.square(prediction_value - target_value)
        return T.mean(mse)
