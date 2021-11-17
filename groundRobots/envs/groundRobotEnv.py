import numpy as np
import time
from abc import abstractmethod

from scipy.integrate import odeint

from gym import core
from gym.utils import seeding


class GroundRobotEnv(core.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

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
        self.viewer = None
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
        self.state = np.zeros(5)
        self.pos_der = np.zeros(3)
        self._dt = dt
        self.seed()
        self._render = render

    @abstractmethod
    def setSpaces(self):
        pass

    def dt(self):
        return self._dt

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, pos=None, vel=None):
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

    def _get_ob(self):
        return np.concatenate((self.state, self.pos_der))

    def _terminal(self):
        return False

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    @abstractmethod
    def integrate(self):
        x0 = self.state
        t = np.arange(0, 2 * self._dt, self._dt)
        ynext = odeint(self.continuous_dynamics, x0, t)
        return ynext[1]

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        s = self.state

        bound_x = self.MAX_POS_BASE + 1.0
        bound_y = self.MAX_POS_BASE + 1.0
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-bound_x, bound_x, -bound_y, bound_y)
        self.viewer.draw_line((-bound_x, 0.0), (bound_x, 0.0))
        self.viewer.draw_line((0.0, -bound_y), (0.0, bound_y))

        if s is None:
            return None

        p = [s[0], s[1]]

        theta = s[2]
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
        time.sleep(self.dt())

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
