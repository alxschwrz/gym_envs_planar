import numpy as np
import time
from abc import abstractmethod
from gym import spaces
from utils.utils import point_inside_circle

from planarenvs.planarCommon.planarEnv import PlanarEnv


class PointRobotEnv(PlanarEnv):

    MAX_VEL = 10
    MAX_POS = 10
    MAX_ACC = 10
    MAX_FOR = 100

    def __init__(self, n=2, dt=0.01, render=False, maxEpisodes=None):
        super().__init__(render=render, dt=dt)
        self._n = n
        if maxEpisodes is None:
            self._maxEpisodes = 10000
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

    #def _reward(self):
    #    reward = -1.0 if not self._terminal() else 0.0
    #    return reward

    def _reward(self):
        reward = 0
        s = self.state
        if point_inside_circle(s['x'][0], s['x'][1], self.sensorState['GoalPosition'][0][0], self.sensorState['GoalPosition'][0][1], self._goals[0].epsilon()):
            # goal reached
            return 100
        if bool(abs(s['x'][0]) > self.MAX_POS or abs(s['x'][1]) > self.MAX_POS or s['x'][1] == 3):
            # out of window
            return -10
        return -0.1

    def _terminal(self):
        s = self.state
        # changed by ALEX
        s = self.state
        is_out_of_window = bool(abs(s['x'][0]) > self.MAX_POS or abs(s['x'][1]) > self.MAX_POS or s['x'][1] == 3)
        goal_pos_reached = point_inside_circle(s['x'][0], s['x'][1], self._goals[0].position()[0],
                                               self._goals[0].position()[1], self._goals[0].epsilon())
        max_episodes_reached = bool(self._t >= self._maxEpisodes)
        # add max episodes
        if is_out_of_window or goal_pos_reached or max_episodes_reached:
            return True
        return False

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
