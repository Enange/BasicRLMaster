from BasicRL.BasicRL import BasicRL
from BasicRL.MyPlotter import MyPlotter
import gym, glob

if __name__ == "__main__":
    print("Hello Basic RL example!")

    # Load Gym Env
    env = gym.make("CartPole-v1", render_mode=None)

    # Run PPO Algorithm
    learner = BasicRL("REINFORCE_PT", gym_env=env, verbose=2, gamma=0.99, exploration_decay=0.99)
    learner.learn(500)

    # Run DQN Algorithm
    learner = BasicRL("REINFORCE", gym_env=env, verbose=2, gamma=0.99, memory_size=10000, exploration_decay=0.99, batch_size=128)
    learner.learn(500)

    # Plot The Results
    plotter = MyPlotter(x_label="Episode", y_label="Reward", title="CartPole v1")
    plotter.load_array([
        glob.glob("data/reward_REINFORCEPT_*.txt"),
        glob.glob("data/reward_REINFORCE_*.txt")
    ])
    plotter.process_data(rolling_window=300, starting_pointer=30)
    plotter.render_std(labels=["REINFORCE_PT", "REINFORCE"], colors=["g", "r"])
    #plotter.render_std(labels=["DQN_PT"], colors=["g"])