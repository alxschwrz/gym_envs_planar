import gym
import planarenvs.pointRobot
import numpy as np


obstacles = False
goal = True
sensors = True


def main():
    env = gym.make("point-robot-vel-v0", render=True, dt=0.01)
    defaultAction = np.array([-0.8, 0.10])
    #defaultAction = lambda t: np.array([np.cos(1.0 * t), np.sin(1.0 * t)])
    initPos = np.array([0.0, -1.0])
    initVel = np.array([-1.0, 0.0])
    n_episodes = 5
    n_steps = 1000
    cumReward = 0.0
    for e in range(n_episodes):
        ob = env.reset(pos=initPos, vel=initVel)

        if sensors:
            from sensors.GoalSensor import GoalSensor
            goalDistObserver = GoalSensor(nbGoals=1, mode='distance')
            env.addSensor(goalDistObserver)
            goalPosObserver = GoalSensor(nbGoals=1, mode='position')
            env.addSensor(goalPosObserver)

        if goal:
            from examples.goal import splineGoal

            env.addGoal(splineGoal)
        print("Starting episode")
        t = 0
        for i in range(n_steps):
            t = t + env.dt()
            action = env.action_space.sample()
            action = defaultAction#(t)
            ob, reward, done, info = env.step(action)
            if (i % 100) == 1:
                print("it: {}, obs: {}, reward: {}".format(i, ob, reward))
                #print(ob, reward, done, info)
            cumReward += reward
            if done:
                break


if __name__ == "__main__":
    main()
