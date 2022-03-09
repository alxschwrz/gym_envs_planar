import gym
import planarenvs.pointRobot
import numpy as np

import yaml
import os
import sys
import forcespro
import gym
import re
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal
from robotmpcs.planner.mpcPlanner import MPCPlanner
import generate_dataset as gd

obstacles = False
goal = True
sensors = True

path_name = os.path.dirname(os.path.realpath(__file__)) + '/'
envMap = {
    'planarArm': 'nLink-reacher-acc-v0',
    'diffDrive': 'ground-robot-acc-v0',
    'pointRobot': 'point-robot-acc-v0',
}

def main():
    env = gym.make("point-robot-vel-v0", render=True, dt=0.01)
    defaultAction = np.array([-0.8, 0.10])
    # defaultAction = lambda t: np.array([np.cos(1.0 * t), np.sin(1.0 * t)])
    initPos = np.array([0.0, -1.0])
    initVel = np.array([-1.0, 0.0])
    n_episodes = 5
    n_steps = 1000
    cumReward = 0.0

    ConfigFile = "/Users/Alex/GIT/testing/robot_mpcs/examples/config/pointRobotMpc.yaml"

    test_setup = os.path.dirname(os.path.realpath(__file__)) + "/" + ConfigFile
    robotType = re.findall('\/(\S*)M', ConfigFile)[0]
    solversDir = os.path.dirname(os.path.realpath(__file__)) + "/solvers/"
    #envName = envMap[robotType]
    recorder = gd.DatasetGenerator(goal=False)
    try:
        myMPCPlanner = MPCPlanner(test_setup, robotType, solversDir)
    except SolverDoesNotExistError as e:
        print(e)
        print("Consider creating it with makeSolver.py")
        return
    myMPCPlanner.concretize()
    n_episodes = 3


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
            action = defaultAction  # (t)
            ob, reward, done, info = env.step(action)
            if (i % 100) == 1:
                print("it: {}, obs: {}, reward: {}".format(i, ob, reward))
                # print(ob, reward, done, info)
            cumReward += reward
            if done:
                break


if __name__ == "__main__":
    main()
