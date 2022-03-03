import numpy as np
from abc import abstractmethod
import time

from planarenvs.planarCommon.planarEnv import PlanarEnv

class MobileBaseEnv(PlanarEnv):

    BASE_HEIGHT = 1.0 # [m]
    BASE_WIDTH = 1.0 # [m]
    LINK_MASS_BASE = 500.0  #: [kg] mass of link 1

    MAX_VEL = 1
    MAX_POS = 5.0
    MAX_ACC = 1.0
    MAX_FOR = 100


    def __init__(self, render=False, dt=0.01):
        super().__init__(render=render, dt=dt)
        self._n = 1
        self._limUpPos = np.ones(self._n) * self.MAX_POS
        self._limUpVel = np.ones(self._n) * self.MAX_VEL
        self._limUpAcc = np.ones(self._n) * self.MAX_ACC
        self._limUpFor = np.ones(self._n) * self.MAX_FOR
        self.setSpaces()

    @abstractmethod
    def setSpaces(self):
        pass

    def _reward(self):
        reward = -1.0 if not self._terminal() else 0.0
        return reward

    def _terminal(self):
        if self.state['x'][0] > self.MAX_POS or self.state['x'][0] < -self.MAX_POS:
            return True
        return False

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def render(self, mode="human"):
        bound = self.MAX_POS + 1.0
        bounds = [bound, bound]
        self.renderCommon(bounds)
        from gym.envs.classic_control import rendering

        # drawAxis
        self.viewer.draw_line((-bound-0.5, 0), (bound+0.5, 0))

        p0 = [self.state['x'][0], 0.5 * self.BASE_HEIGHT]
        tf = rendering.Transform(rotation=0, translation=p0)
        l, r, t, b = -0.5 * self.BASE_WIDTH, 0.5 * self.BASE_WIDTH, 0.5 * self.BASE_HEIGHT, -0.5 * self.BASE_HEIGHT
        link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        link.set_color(0, 0.8, 0.8)
        link.add_attr(tf)
        time.sleep(self.dt())

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
