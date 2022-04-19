import numpy as np
import time
from abc import abstractmethod
from gym import spaces
from utils.utils import point_inside_circle

from planarenvs.planarCommon.planarEnv import PlanarEnv


class PointRobotEnv(PlanarEnv):

    MAX_VEL = 1  # necessary for stable training
    MAX_POS = 10
    MAX_ACC = 10
    MAX_FOR = 100

    def __init__(self, n=2, dt=0.01, render=False, maxEpisodes=10000, goalSamplingDistRatio=1):
        super().__init__(render=render, dt=dt)
        self._lastGoalReached = False
        self._n = n
        self._maxEpisodes = maxEpisodes
        self._goalSamplingDistRatio = goalSamplingDistRatio
        self._stepsGoalReached = 0
        self._lastStepGoalReached = False

        self._limits = {
            'pos': {'high': np.ones(self._n) * self.MAX_POS, 'low': np.ones(self._n) * -self.MAX_POS},
            'vel': {'high': np.ones(self._n) * self.MAX_VEL, 'low': np.ones(self._n) * -self.MAX_VEL},
            'acc': {'high': np.ones(self._n) * self.MAX_ACC, 'low': np.ones(self._n) * -self.MAX_ACC},
            'for': {'high': np.ones(self._n) * self.MAX_FOR, 'low': np.ones(self._n) * -self.MAX_FOR},
        }
        self._limUpPos = self._limits['pos']['high']
        self._limUpVel = self._limits['vel']['high']
        self._limUpAcc = self._limits['acc']['high']
        self._limUpFor = self._limits['for']['high']
        self.setSpaces()

        from sensors.GoalSensor import GoalSensor
        goalDistObserver = GoalSensor(nbGoals=1, mode='distance')
        self.addSensor(goalDistObserver)

    def resetLimits(self, **kwargs):
        for key in (kwargs.keys() & self._limits.keys()):
            limitCandidate = kwargs.get(key)
            for limit in (['low', 'high'] & limitCandidate.keys()):
                if limitCandidate[limit].size == self._n:
                    self._limits[key][limit] = limitCandidate[limit]
                else:
                    import logging
                    logging.warning("Ignored reset of limit because the size of the limit is incorrect.")

        self._limUpPos = self._limits['pos']['high']
        self._limUpVel = self._limits['vel']['high']
        self._limUpAcc = self._limits['acc']['high']
        self._limUpFor = self._limits['for']['high']
        self.observation_space.spaces['x'] = spaces.Box(low=-self._limUpPos, high=self._limUpPos, dtype=np.float64)
        self.observation_space.spaces['xdot'] = spaces.Box(low=-self._limUpVel, high=self._limUpVel, dtype=np.float64)


    @abstractmethod
    def setSpaces(self):
        pass

    def _reward(self):
        # todo: Reward function needs big rework:
        # todo: shaped reward
        # todo: removing sensorState from the equation -> done
        reward = 0
        s = self.state

        # Reward Weights
        W_GOAL = 1
        W_OBST = -1
        W_DIST = -1
        W_STEP = 0

        # robot inside goal region
        if point_inside_circle(s['x'][0], s['x'][1], self._goals[0].position()[0],
                               self._goals[0].position()[1], self._goals[0].epsilon()):
            reward = reward + W_GOAL

        # robot out of window
        if bool(abs(s['x'][0]) >= self.MAX_POS or abs(s['x'][1]) >= self.MAX_POS):
            reward = reward + W_OBST

        # L2 distance weighted reward
        initialDist = np.sqrt(np.sum(np.power(self._goals[0].position(), 2)))
        currDist = np.sqrt(np.sum(np.power(self._goals[0].position() - s['x'], 2)))
        reward = reward + W_DIST * (currDist / initialDist)

        return reward + W_STEP

    def _terminal(self):
        s = self.state
        is_out_of_window = bool(abs(s['x'][0]) >= self.MAX_POS or abs(s['x'][1]) >= self.MAX_POS)
        goal_pos_reached = point_inside_circle(s['x'][0], s['x'][1], self._goals[0].position()[0],
                                               self._goals[0].position()[1], self._goals[0].epsilon())
        max_episodes_reached = bool(self._t >= self._maxEpisodes * self._dt)

        if goal_pos_reached and self._lastStepGoalReached:
            self._stepsGoalReached += 1
            self._lastStepGoalReached = True
        else:
            self._stepsGoalReached = 0

        self._lastStepGoalReached = bool(goal_pos_reached)

        if max_episodes_reached or self._stepsGoalReached >= 1000:
            return True
        return False

    def reset(self, pos=None, vel=None):
        ## new reset method, that configures experiment.
        # this config can totally setup the whole configuration.
        self.resetCommon()
        if not isinstance(pos, np.ndarray) or not pos.size == self._n:
            pos = np.array([round(np.random.uniform(low=-self._limUpPos[0], high=self._limUpPos[0]), 2),
                            round(np.random.uniform(low=-self._limUpPos[0], high=self._limUpPos[0]), 2)])
            pos = np.zeros(self._n)
        if not isinstance(vel, np.ndarray) or not vel.size == self._n:
            vel = np.array([round(np.random.uniform(low=-self._limUpVel[0], high=self._limUpVel[0]), 2),
                            round(np.random.uniform(low=-self._limUpVel[0], high=self._limUpVel[0]), 2)])
            vel = np.zeros(self._n)
        self.state = {'x': pos, 'xdot': vel}

        # setting new goal
        from MotionPlanningGoal.staticSubGoal import StaticSubGoal
        rat = self._goalSamplingDistRatio
        goalPos = [round(np.random.uniform(low=-self._limUpPos[0]*rat, high=self._limUpPos[0]*rat), 2),
                   round(np.random.uniform(low=-self._limUpPos[0]*rat, high=self._limUpPos[0]*rat), 2)]
        # goalPos = [7, 2]
        staticGoalDict = {
            "m": 2, "w": 1.0, "prime": True, 'indices': [0, 1], 'parent_link': 0, 'child_link': 3,
            'desired_position': goalPos, 'epsilon': 0.5, 'type': "staticSubGoal",
        }
        staticGoal = StaticSubGoal(name="goal1", contentDict=staticGoalDict)
        self.addGoal(staticGoal)
        # todo: remove this hardcoded sensorState reset.
        resetSensorState = {}
        for sensor in self._sensors:
            # resetSensorState[sensor.name()] = np.zeros(sensor.getOSpaceSize())
            resetSensorState[sensor.name()] = sensor.sense(self.state, self._goals, self._obsts, self.t())
        self.sensorState = resetSensorState
        self._lastStepGoalReached = 0
        self._lastStepGoalReached = False
        return self._get_ob()

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def render(self, mode="human"):
        self.renderCommon(self._limits)
        from gym.envs.classic_control import rendering

        # drawAxis
        self.viewer.draw_line((self._limits['pos']['low'][0], 0), (self._limits['pos']['high'][0], 0))
        self.viewer.draw_line((0, self._limits['pos']['low'][1]), (0, self._limits['pos']['high'][1]))
        # drawPoint
        x = self.state['x'][0:2]
        tf0 = rendering.Transform(rotation=0, translation=(x[0], x[1]))
        joint = self.viewer.draw_circle(.10)
        joint.set_color(.8, .8, 0)
        joint.add_attr(tf0)
        time.sleep(self.dt())

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
