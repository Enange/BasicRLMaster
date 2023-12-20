import glob
import gym

from BasicRL.BasicRL import BasicRL
from BasicRL.MyPlotter import MyPlotter

if __name__ == "__main__":
    print("Hello Basic RL example!")

    # Load Gym Env
    #env = gym.make("Pendulum-v1", render_mode=None, g=9.81)
    env = gym.make("MountainCarContinuous-v0")

    # # Run PPO Algorithm
    # learner = BasicRL("REINFORCE_PT", gym_env=env, verbose=2, gamma=0.99, exploration_decay=0.99)
    # learner.learn(1000)
    #
    # # # Run PPO Algorithm
    # # learner = BasicRL("REINFORCE_PT", gym_env=env, verbose=2, gamma=0.99, exploration_decay=0.99, sigma=2.0)
    # # learner.learn(1000)
    #
    # # Run DQN Algorithm
    # learner = BasicRL("DQN_PT", gym_env=env, verbose=2, gamma=0.99, memory_size=10000, exploration_decay=0.99,
    #                   batch_size=128)
    # learner.learn(1000)

    # Plot The Results
    plotter = MyPlotter(x_label="Episode", y_label="Reward", title="MountainCarContinuous")
    plotter.load_array([
        glob.glob("data/reward_REINFORCEPT_sigma_*.txt"),
        glob.glob("data/reward_REINFORCEPT_gamma_*.txt"),
        glob.glob("data/reward_REINFORCEPT_hidden32_*.txt")
    ])
    plotter.process_data(rolling_window=300, starting_pointer=30)
    plotter.render_std(labels=["R_Sigma(2)", "R_Gamma(0.99)", "R_hidden(32)"], colors=["g", "r", "b"])
    #plotter.render_std(labels=["REINFORCE_PT"], colors=["g"])
