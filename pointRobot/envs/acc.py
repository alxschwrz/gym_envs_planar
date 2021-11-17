import numpy as np
from numpy import sin, cos, pi
import time

from scipy.integrate import odeint

from gym import core, spaces
from gym.utils import seeding

from utils.utils import point_inside_circle


class PointRobotAccEnv(core.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

    MAX_VEL = 8
    MAX_POS = 5
    MAX_ACC = 9

    def __init__(self, render=False, n=2, dt=0.01, initPos=None, initVel=None, goalPos=None, goalSize=None):
        if initVel is None:
            initVel = [0, 0]
        if initPos is None:
            initPos = [0, 0]
        if goalPos is None:
            goalPos = [0, 0]
        if goalSize is None:
            goalSize = 1

        self._n = n
        self.viewer = None
        self._initPos = initPos  # added by alex
        self._initVel = initVel  # added by alex
        self._goalPos = goalPos  # added by alex
        self._goalSize = goalSize  # added by alex
        limUpPos = [self.MAX_POS for i in range(n)]
        limUpVel = [self.MAX_VEL for i in range(n)]
        limUpAcc = [self.MAX_ACC for i in range(n)]
        high = np.array(limUpPos + limUpVel,  dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)
        self.action_space = spaces.Box(
            low=-np.array(limUpAcc), high=np.array(limUpAcc), dtype=np.float64
        )
        self.state = None
        self.seed()
        self._dt = dt
        self._render = render

    def dt(self):
        return self._dt

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = np.concatenate((self._initPos, self._initVel))
        return self._get_ob()

    def step(self, a):
        s = self.state
        self.action = a
        ns = self.integrate()
        self.state = ns
        terminal = self._terminal()

        ## reward function
        reward = self.reward()

        if self._render:
            self.render()
        return (self._get_ob(), reward, terminal, {})

    def reward(self):
        reward = 0
        s = self.state
        if point_inside_circle(s[0], s[1], self._goalPos[0], self._goalPos[1], self._goalSize):
            # goal reached
            return 100
        if bool(abs(s[0]) > self.MAX_POS or abs(s[1]) > self.MAX_POS or s[1] == 3):
            # out of window
            return -10
        return -0.1

    def _get_ob(self):
        return self.state

    def _terminal(self):
        # changed by ALEX
        s = self.state
        is_out_of_window = bool(abs(s[0]) > self.MAX_POS or abs(s[1]) > self.MAX_POS or s[1] == 3)
        goal_pos_reached = point_inside_circle(s[0], s[1], self._goalPos[0], self._goalPos[1], self._goalSize)
        if is_out_of_window or goal_pos_reached:
            return True
        return False


    def continuous_dynamics(self, x, t):
        u = self.action
        vel = np.array(x[self._n: self._n * 2])
        acc = np.concatenate((vel, u))
        return acc

    def integrate(self):
        x0 = self.state[0:2 * self._n]
        t = np.arange(0, 2 * self._dt, self._dt)
        acc = self.continuous_dynamics(x0, t)
        ynext = x0 + self._dt * acc
        return ynext

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        s = self.state
        if s is None:
            return None

        bound = 5.0
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        self.viewer.draw_line((-bound, 0), (bound, 0))
        self.viewer.draw_line((0, -bound), (0, bound))
        x = s[0]
        y = 0.0
        if self._n == 2:
            y = s[1]
        tf0 = rendering.Transform(rotation=0, translation=(x, y))
        joint = self.viewer.draw_circle(.10)
        joint.set_color(.8, .8, 0)
        joint.add_attr(tf0)

        # rendering for goal circle
        tfg = rendering.Transform(rotation=0, translation=(self._goalPos[0], self._goalPos[1]))
        goal = self.viewer.draw_circle(self._goalSize)
        goal.set_color(0, 128, 0)
        goal.add_attr(tfg)

        time.sleep(self.dt())

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None