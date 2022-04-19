import gym
import planarenvs.pointRobot
import numpy as np
from pynput import keyboard
from examples.goal import staticGoal


def main():
    env = gym.make("point-robot-vel-v0", render=True, dt=0.01, goalSamplingDistRatio=0.5)

    n_episodes = 20
    n_steps = 1000
    cumReward = 0.0

    for e in range(n_episodes):
        # ob = env.reset(pos=initPos, vel=initVel)
        ob = env.reset()

        print("Starting episode")
        t = 0
        for i in range(n_steps):
            t = t + env.dt()

            defaultAction = np.array([2, 0.50])
            if i > 350:
                defaultAction = np.array([0, 0.1])
            #keyboardAction = np.array([0.1, 0.1])
            # action = defaultAction  # (t)

            ob, reward, done, info = env.step(defaultAction)
            if True:#(i % 20) == 1:
                print("it: {}, act: {}, obs: {}, done: {}, reward: {}".format(i, defaultAction, ob, done, reward))
                # print(ob, reward, done, info)
            cumReward += reward
            if done:
                break


if __name__ == "__main__":
    main()
