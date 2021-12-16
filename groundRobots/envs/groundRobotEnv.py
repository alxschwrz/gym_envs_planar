import numpy as np
import time
from scipy.integrate import odeint
from abc import abstractmethod

from planarCommon.planarEnv import PlanarEnv


class GroundRobotEnv(PlanarEnv):

    BASE_WIDTH = 1.0  # [m]
    BASE_LENGTH = 1.3  # [m]
    BASE_WHEEL_DIST = 0.6  # [m]
    LINK_MASS_BASE = 5.0  #: [kg] mass of link 1

    MAX_POS_BASE = 5
    MAX_POS_BASE_THETA = np.pi
    MAX_VEL_BASE = 5
    MAX_VEL_BASE_THETA = 5
    MAX_ACC_BASE = 100
    MAX_ACC_BASE_THETA = 100
    MAX_VEL_FORWARD = 1.0
    MAX_ACC_FORWARD = 100

    def __init__(self, render=False, dt=0.01):
        super().__init__(render=render, dt=dt)
        self._limUpPos = np.array(
            [self.MAX_POS_BASE, self.MAX_POS_BASE, self.MAX_POS_BASE_THETA]
        )
        self._limUpVel = np.array(
            [self.MAX_VEL_BASE, self.MAX_VEL_BASE, self.MAX_VEL_BASE_THETA]
        )
        self._limUpRelVel = np.array([self.MAX_VEL_FORWARD, self.MAX_VEL_BASE_THETA])
        self._limUpAcc = np.array(
            [self.MAX_ACC_BASE, self.MAX_ACC_BASE, self.MAX_ACC_BASE_THETA]
        )
        self._limUpRelAcc = np.array([self.MAX_ACC_FORWARD, self.MAX_ACC_BASE_THETA])
        self.setSpaces()
        self.pos_der = np.zeros(3)

    @abstractmethod
    def setSpaces(self):
        pass

    def reset(self, pos=None, vel=None):
        self.resetCommon()
        """ The velocity is the forward velocity and turning velocity here """
        if not isinstance(pos, np.ndarray) or not pos.size == 3:
            pos = np.zeros(3)
        if not isinstance(vel, np.ndarray) or not vel.size == 2:
            vel = np.zeros(2)
        self.state = np.concatenate((pos, vel))
        return self._get_ob()

    def step(self, a):
        s = self.state
        self.action = a
        _ = self.continuous_dynamics(s, 0)
        ns = self.integrate()
        self.state = ns
        terminal = self._terminal()
        reward = -1.0 if not terminal else 0.0
        if self._render:
            self.render()
        return (self._get_ob(), reward, terminal, {})

    def integrate(self):
        self._t += self.dt()
        x0 = self.state
        t = np.arange(0, 2 * self._dt, self._dt)
        ynext = odeint(self.continuous_dynamics, x0, t)
        return ynext[1]

    def _get_ob(self):
        return np.concatenate((self.state, self.pos_der))

    def _terminal(self):
        return False

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def render(self, mode="human", final=True):
        bound_x = self.MAX_POS_BASE + 1.0
        bound_y = self.MAX_POS_BASE + 1.0
        bounds = [bound_x, bound_y]
        self.renderCommon(bounds)
        from gym.envs.classic_control import rendering

        # drawAxis
        self.viewer.draw_line((-bound_x, 0.0), (bound_x, 0.0))
        self.viewer.draw_line((0.0, -bound_y), (0.0, bound_y))

        p = [self.state[0], self.state[1]]

        theta = self.state[2]
        tf = rendering.Transform(rotation=theta, translation=p)

        l, r, t, b = (
            -0.5 * self.BASE_LENGTH,
            0.5 * self.BASE_LENGTH,
            0.5 * self.BASE_WIDTH,
            -0.5 * self.BASE_WIDTH,
        )
        link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        yw = self.BASE_WHEEL_DIST / 2.0
        wheelfl = self.viewer.draw_polygon(
            [(0.2, yw), (0.2, yw + 0.1), (0.4, yw + 0.1), (0.4, yw)]
        )
        wheelfr = self.viewer.draw_polygon(
            [(0.2, -yw), (0.2, -yw - 0.1), (0.4, -yw - 0.1), (0.4, -yw)]
        )
        wheelbl = self.viewer.draw_polygon(
            [(-0.2, yw), (-0.2, yw + 0.1), (-0.4, yw + 0.1), (-0.4, yw)]
        )
        wheelbr = self.viewer.draw_polygon(
            [(-0.2, -yw), (-0.2, -yw - 0.1), (-0.4, -yw - 0.1), (-0.4, -yw)]
        )
        wheelfl.add_attr(tf)
        wheelfr.add_attr(tf)
        wheelbl.add_attr(tf)
        wheelbr.add_attr(tf)
        link.set_color(0, 0.8, 0.8)
        link.add_attr(tf)

        if final:
            time.sleep(self.dt())
            return self.viewer.render(return_rgb_array=mode == "rgb_array")
