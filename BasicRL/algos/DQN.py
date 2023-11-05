from tensorflow import keras
from collections import deque
import tensorflow as tf
import numpy as np
import gym
import random


######
# Riga 102
# Capire il metodo actor objective function double
#
# ####

class DQN:
    # COSTRUTTORE
    def __init__(self, env, verbose):
        # Prende enviroment da Gym, Verbose mostra l'avanzamento
        self.env = env
        self.verbose = verbose

        self.input_shape = self.env.observation_space.shape  # Input possibili
        self.action_space = env.action_space.n  # Output possibili
        self.actor = self.get_actor_model(self.input_shape, self.action_space) #Ritorna un modello neurale dati input e output

        self.actor_target = self.get_actor_model(self.input_shape, self.action_space)
        self.actor_target.set_weights(self.actor.get_weights())                             # PRENDE I PESII ???????

        self.optimizer = keras.optimizers.Adam()    # OTTIMIZZA
        self.gamma = 0.95   # Ammortamento Premi
        self.memory_size = 2000 #Dimensione Memoria
        self.batch_size = 32    #Numeri campioni propagati nella rete
        self.exploration_rate = 1.0 #Tasso iniziale di Exploration
        self.exploration_decay = 0.995 # Fattore di decadimento
        self.tau = 0.005    #?????

        self.run_id = np.random.randint(0, 1000) # ritorna un numero intero tra una distribuzione discreta per salvare i dati /data/...
        self.render = False # Render ma non sappiamo se serve ???

    def loop(self, num_episodes=1000):
        reward_list = []
        ep_reward_mean = deque(maxlen=100) # usato per i dati
        replay_buffer = deque(maxlen=self.memory_size) # lista per salvare [state, action, reward, new_state, done]

        # ciclo per tutti gli episodi (in example)
        for episode in range(num_episodes):
            state, info = self.env.reset(seed=123, options={})
            ep_reward = 0   # REset reward ad ogni tentativo

            while True:
                if self.render: self.env.render()
                action = self.get_action(state)     # Ottengo l'azione da fare
                new_state, reward, terminated, truncated, _ = self.env.step(action) #Vedo il risultato dell'azione nell' enviroment
                ep_reward += reward     # Somma dei rewards dell'episodio
                done = terminated or truncated  # Mi rendo conto se l'azione è fallita o terminata -> rifaccio partire il mio enviroment

                replay_buffer.append([state, action, reward, new_state, done])# Aggiorno replay buffer( record dei [state, action, reward, new_state, done])

                # Episodio finito
                if done: break
                state = new_state       # Aggiorno lo stato in cui sono attualmente

                self.update_networks(replay_buffer)
                self._update_target(self.actor.variables, self.actor_target.variables, tau=self.tau)

            self.exploration_rate = self.exploration_rate * self.exploration_decay if self.exploration_rate > 0.05 else 0.05        # Aggiorno exploraion rate
            ep_reward_mean.append(ep_reward)    # Ridondante
            reward_list.append(ep_reward)       # salvo i miei punteggi
            # Salvo i miei dati
            if self.verbose > 0: print(
                f"Episode: {episode:7.0f}, reward: {ep_reward:8.2f}, mean_last_100: {np.mean(ep_reward_mean):8.2f}, exploration: {self.exploration_rate:0.2f}")
            if self.verbose > 1: np.savetxt(f"data/reward_DQN_{self.run_id}.txt", reward_list)

    def _update_target(self, weights, target_weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def get_action(self, state):
        # Azione randomica all'inizio sarà sicuramente randomica
        # exploration rate = 1 e 0 <= random <= 1
        # pian piano si abbassa l'exploration rate e non farà più azioni casuali
        if np.random.random() < self.exploration_rate:
            return np.random.choice(self.action_space) # a Caso dalle scelte
        return np.argmax(self.actor(state.reshape((1, -1)))) # Scelta data dalla rete neurale

    def update_networks(self, replay_buffer):
        samples = np.array(random.sample(replay_buffer, min(len(replay_buffer), self.batch_size)), dtype=object)  # Prendo un Campione per la mia rete neurale
        with tf.GradientTape() as tape:         # file = open()
            objective_function = self.actor_objective_function_double(samples)  # Compute loss with custom loss function

            print ("Objective function:", objective_function)
            grads = tape.gradient(objective_function,
                                  self.actor.trainable_variables)  # Compute gradients actor for network
            self.optimizer.apply_gradients(
                zip(grads, self.actor.trainable_variables))  # Apply gradients to update network weights

    def actor_objective_function_double(self, replay_buffer):
        state = np.vstack(replay_buffer[:, 0]) # Prende dal RB lo stato
        action = replay_buffer[:, 1] # Prende dal RB l'azione
        reward = np.vstack(replay_buffer[:, 2]) # Prende dal RB il reward
        new_state = np.vstack(replay_buffer[:, 3]) # Prende dal RB il nuovo stato
        done = np.vstack(replay_buffer[:, 4]) # Prende dal RB il done

        next_state_action = np.argmax(self.actor(new_state), axis=1)    #Calcolo le prossime azioni e ritorna 0 e 1
        target_mask = self.actor_target(new_state) * tf.one_hot(next_state_action, self.action_space)
        target_mask = tf.reduce_sum(target_mask, axis=1, keepdims=True)     # Sommando ogni riga, in un valore
        print(type(target_mask))

        target_value = reward + (1 - done.astype(int)) * self.gamma * target_mask
        mask = self.actor(state) * tf.one_hot(action, self.action_space)
        prediction_value = tf.reduce_sum(mask, axis=1, keepdims=True)

        mse = tf.math.square(prediction_value - target_value)
        return tf.math.reduce_mean(mse)



    def get_actor_model(self, input_shape, output_size):
        inputs = keras.layers.Input(shape=input_shape)
        hidden_0 = keras.layers.Dense(64, activation='relu')(inputs)
        hidden_1 = keras.layers.Dense(64, activation='relu')(hidden_0)
        outputs = keras.layers.Dense(output_size, activation='linear')(hidden_1)

        return keras.Model(inputs, outputs)

    # def get_actor_modelPT(self, input_shape, output_size):
    # 	    self.layer_1 = nn.Linear(in_features=input_shape, out_features= 64)
    #     self.layer_2 = nn.Linear(in_features=self.layer_1, out_features= 64)
    #     self.layer_3 = nn.Linear(in_features=self.layer_2, out_features= output_size)
    #     self.relu = nn.ReLU()
    #     return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

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
