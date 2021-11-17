import gym
import pointRobot
import numpy as np
from stable_baselines3 import PPO
from utils.utils import point_inside_circle


def main():
    env = gym.make('point-robot-acc-v0', render=False, dt=0.01, goalPos=[5, 3], goalSize=0.5)
    eval_env = gym.make('point-robot-acc-v0', render=True, dt=0.01, goalPos=[5, 3], goalSize=0.5)

    defaultAction = [1.0, 0.0]
    initPos = np.array([1.0, 0.0])
    initVel = np.array([0.0, 0.0])
    n_episodes = 1
    n_steps = 500000
    cumReward = 0.0

    #model = PPO('MlpPolicy', env, verbose=1)
    #model.learn(total_timesteps=n_steps, log_interval=4)
    #model.save("models/ppo_point_1")
    #env = model.get_env()
    #del model
    model = PPO.load('models/ppo_point_1')

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