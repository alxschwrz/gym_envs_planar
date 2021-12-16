import numpy as np
import time
from abc import abstractmethod

from planarCommon.planarEnv import PlanarEnv

class PointRobotEnv(PlanarEnv):

    MAX_VEL = 10
    MAX_POS = 10
    MAX_ACC = 10
    MAX_FOR = 100

    def __init__(self, n=2, dt=0.01, render=False):
        super().__init__(render=render, dt=dt)
        self._n = n
        self._limUpPos = np.ones(self._n) * self.MAX_POS
        self._limUpVel = np.ones(self._n) * self.MAX_VEL
        self._limUpAcc = np.ones(self._n) * self.MAX_ACC
        self._limUpFor = np.ones(self._n) * self.MAX_FOR
        self.setSpaces()

    @abstractmethod
    def setSpaces(self):
        pass

    def step(self, a):
        s = self.state
        self.action = a
        _ = self.continuous_dynamics(self.state, 0)
        ns = self.integrate()
        self.state = ns
        terminal = self._terminal()
        reward = -1.0 if not terminal else 0.0
        if self._render:
            self.render()
        return (self._get_ob(), reward, terminal, {})

    def _get_ob(self):
        return self.state

    def _terminal(self):
        s = self.state
        return False

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def render(self, mode="human"):
        bounds = [5, 5]
        self.renderCommon(bounds)
        from gym.envs.classic_control import rendering

        # drawAxis
        self.viewer.draw_line((-bounds[0], 0), (bounds[0], 0))
        self.viewer.draw_line((0, -bounds[1]), (0, bounds[1]))
        # drawPoint
        x = self.state[0:2]
        tf0 = rendering.Transform(rotation=0, translation=(x[0], x[1]))
        joint = self.viewer.draw_circle(.10)
        joint.set_color(.8, .8, 0)
        joint.add_attr(tf0)
        time.sleep(self.dt())

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
