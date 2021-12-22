import gym
import pointRobot
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import SAC
from utils.utils import point_inside_circle


def main():

    goalSize = 0.5
    env = gym.make('point-robot-vel-v0', render=False, dt=0.01, goalSize=goalSize)
    eval_env = gym.make('point-robot-vel-v0', render=True, dt=0.01, goalSize=goalSize)

    defaultAction = [1.0, 0.0]
    initPos = np.array([1.0, 0.0])
    initVel = np.array([0.0, 0.0])
    n_episodes = 10
    n_steps = 5000000
    cumReward = 0.0
    env.reset(goal_soze=4)
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="tensorboard/", )

    #model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log="tensorboard/")
    #model = SAC('MlpPolicy', env, verbose=1, tensorboard_log="tensorboard/", device='cuda')


    #model = PPO('MultiInputPolicy', env, verbose=1) # change to multiinputpolicy
    model.learn(total_timesteps=n_steps, log_interval=4, tb_log_name="PPO_relGoal_size{}_vel".format(goalSize))
    #model.save("models/PPO_rel_goal_size{}".format(goalSize))
    #env = model.get_env()
    del model
    model = PPO.load('models/PPO_rel_goal_size{}_vel'.format(goalSize))
    #model = SAC.load('models/SAC_point_goal_cond_1.zip')
    for e in range(n_episodes):
        ob = eval_env.reset()
        print("Starting episode")
        for i in range(n_steps):
            action, _states = model.predict(ob, deterministic=True)
            #action = env.action_space.sample()
            #action = defaultAction
            ob, reward, done, info = eval_env.step(action)
            cumReward += reward
            if done:
                break


if __name__ == '__main__':
    main()
