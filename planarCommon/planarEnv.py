from abc import abstractmethod
import numpy as np
from scipy.integrate import odeint

from gym import core
from gym.utils import seeding


class PlanarEnv(core.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

    def __init__(self, render=False, dt=0.01):
        self.viewer = None
        self.state = None
        self.seed()
        self._dt = dt
        self._t = 0.0
        self._render = render
        self._obsts = []
        self._goals = []

    @abstractmethod
    def setSpaces(self):
        pass

    def dt(self):
        return self._dt

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def addObstacle(self, obst):
        self._obsts.append(obst)

    def addGoal(self, goal):
        self._goals.append(goal)

    def t(self):
        return self._t

    def resetCommon(self):
        self._obsts = []
        self._goals = []
        self._t = 0.0

    def reset(self, pos=None, vel=None):
        self.resetCommon()
        if not isinstance(pos, np.ndarray) or not pos.size == self._n:
            pos = np.zeros(self._n)
        if not isinstance(vel, np.ndarray) or not vel.size == self._n:
            vel = np.zeros(self._n)
        self.state = np.concatenate((pos, vel))
        return self._get_ob()

    @abstractmethod
    def step(self, a):
        pass

    @abstractmethod
    def _get_ob(self):
        pass

    @abstractmethod
    def _terminal(self):
        pass

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def integrate(self):
        self._t += self.dt()
        x0 = self.state[0 : 2 * self._n]
        t = np.arange(0, 2 * self._dt, self._dt)
        ynext = odeint(self.continuous_dynamics, x0, t)
        return ynext[1]

    @abstractmethod
    def render(self, mode="human"):
        pass

    def renderCommon(self, bounds):
        from gym.envs.classic_control import rendering

        if self.state is None:
            return None
        if self.viewer is None:
            if isinstance(bounds, list):
                self.viewer = rendering.Viewer(500, 500)
                self.viewer.set_bounds(-bounds[0], bounds[1], -bounds[1], bounds[1])
            elif isinstance(bounds, dict):
                ratio = (bounds["pos"]["high"][0] - bounds["pos"]["low"][0]) / (
                    bounds["pos"]["high"][1] - bounds["pos"]["low"][1]
                )
                if ratio > 1:
                    windowSize = (1000, int(1000 / ratio))
                else:
                    windowSize = (int(ratio * 1000), 1000)
                self.viewer = rendering.Viewer(windowSize[0], windowSize[1])
                self.viewer.set_bounds(
                    bounds["pos"]["low"][0],
                    bounds["pos"]["high"][0],
                    bounds["pos"]["low"][1],
                    bounds["pos"]["high"][1],
                )
        for obst in self._obsts:
            obst.renderGym(self.viewer, t=self.t())
        for goal in self._goals:
            goal.renderGym(self.viewer, t=self.t())

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
