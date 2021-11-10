import numpy as np
from numpy import sin, cos, pi
import time

from scipy.integrate import odeint

from gym import core, spaces
from gym.utils import seeding


class NLinkVelReacherEnv(core.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

    LINK_LENGTH = 1.0  # [m]

    MAX_VEL = 4 * pi
    MAX_POS = pi

    def __init__(self, render=False, n=2, dt=0.01):
        self._n = n
        self.viewer = None
        limUpPos = [self.MAX_POS for i in range(n)]
        limUpVel = [self.MAX_VEL for i in range(n)]
        high = np.array( limUpPos + limUpVel, dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)
        self.action_space = spaces.Box(
            low=-np.array(limUpVel), high=np.array(limUpVel), dtype=np.float64
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

    def reset(self, pos=None, vel=None):
        try:
            if pos==None:
                pos=np.ones(self._n) * 0.1
            if vel==None:
                vel=np.ones(self._n) * 0.1
        except:
            print("Using initial data")
        self.state = np.concatenate((pos, vel))
        return self._get_ob()

    def step(self, a):
        s = self.state
        self.action = a
        ns = self.integrate()
        vel = a
        self.state = np.concatenate((ns, vel))
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

    def continuous_dynamics(self, x, t):
        vel = self.action
        return vel

    def integrate(self):
        x0 = self.state[0:self._n]
        t = np.arange(0, 2 * self._dt, self._dt)
        ynext = odeint(self.continuous_dynamics, x0, t)
        return ynext[1]

    def forwardKinematics(self, lastLinkIndex):
        fk = np.array([0.0, 0.2, 0.0])
        for i in range(lastLinkIndex):
            angle = 0.0
            for j in range(i+1):
                angle += self.state[j]
            fk[0] += np.cos(angle) * self.LINK_LENGTH
            fk[1] += np.sin(angle) * self.LINK_LENGTH
            fk[2] += self.state[i]
        fk[2] += self.state[lastLinkIndex]
        return fk

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        s = self.state

        bound = self.LINK_LENGTH * self._n + 0.2
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        if s is None:
            return None

        self.viewer.draw_line((-bound, 0), (bound, 0))
        base = self.viewer.draw_polygon([(-0.2,0), (0.0,0.2), (0.2,0), (-0.2,0)])
        baseJoint = self.viewer.draw_circle(.10)
        baseJoint.set_color(.8, .8, 0)
        tf0 = rendering.Transform(rotation=0, translation=(0.0, 0.2))
        baseJoint.add_attr(tf0)
        l,r,t,b = 0, self.LINK_LENGTH, .01, -.01
        for i in range(self._n):
            fk = self.forwardKinematics(i)
            tf = rendering.Transform(rotation=fk[2], translation=fk[0:2])
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.set_color(0,.8, .8)
            link.add_attr(tf)
            joint = self.viewer.draw_circle(.10)
            joint.set_color(.8, .8, 0)
            joint.add_attr(tf)
        fk = self.forwardKinematics(self._n)
        tf = rendering.Transform(rotation=fk[2], translation=fk[0:2])
        eejoint = self.viewer.draw_circle(.10)
        eejoint.set_color(.8, .8, 0)
        eejoint.add_attr(tf)
        time.sleep(self.dt())


        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
