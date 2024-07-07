import glob
import gym
import gymnasium
import torch

#from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


from BasicRL.BasicRL import BasicRL
from BasicRL.MyPlotter import MyPlotter
from MaplessNav import MaplessNav
from wrappers import MultiCostWrapper

if __name__ == "__main__":
    print("Hello Basic RL example!")

    # Load Gym Env
    # env = gymnasium.make("BipedalWalker-v3")

    #env = gym.make("MountainCarContinuous-v0")
    env = MaplessNav(render_mode=None)
    env = MultiCostWrapper(env)
    env = gymnasium.wrappers.FlattenObservation(env)
    env = gymnasium.wrappers.ClipAction(env)
    observation = env.reset()

    # model.learn(total_timesteps=100000)
    # model.save("ppo_robot")
    #
    # del model
    #
    # model = PPO.load("ppo_robot")
    # obs = env.reset()
    # while True:
    #     action, _states = model.load()
    #     obs, rewards, dones, info = env.step(action)
    #     env.render("human")
    #Run PPO Algorithm
    # for i in range(4):
    #     learner = BasicRL("PPO_PT", gym_env=env, verbose=2)
    #     learner.learn(1000)
    # for i in range(4):
    #     #Run PPO Algorithm
    #     learner = BasicRL("PPO_PT", gym_env=env, verbose=2)
    #     learner.learn(10000)
    #     learner = BasicRL("REINFORCE_PT", gym_env=env, verbose=2)
    #     learner.learn(1000)

    # # Run DQN Algorithm
    # learner = BasicRL("PPO_PT", gym_env=env, verbose=1)
    # learner.learn(1000)

    # Plot The Results
    plotter = MyPlotter(x_label="Episode", y_label="Collision", title="PPO Collision PyGame")
    plotter.load_array([
        glob.glob("data/PyGame/PPO*_10kC.txt"),
        #glob.glob("data/PyGame/DDPG*_success.txt"),
        #glob.glob("data/PyGame/REINFORCE*_success.txt")
    ])
    plotter.process_data(rolling_window=300, starting_pointer=30)
    plotter.render_std(labels=["PPO"], colors=["b", "r", "g"])
    # plotter.render_std(labels=["REINFORCE_PT"], colors=["g"])

