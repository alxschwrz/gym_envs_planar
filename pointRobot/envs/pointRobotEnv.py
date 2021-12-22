import numpy as np
import time
from abc import abstractmethod

from scipy.integrate import odeint

from gym import core
from gym.utils import seeding
from gym import spaces

from utils.utils import point_inside_circle

class PointRobotEnv(core.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

    MAX_VEL = 10
    MAX_POS = 10
    MAX_ACC = 10
    MAX_FOR = 100

    def __init__(self, n=2, dt=0.01, render=False, initPos=None, initVel=None, goalPos=None, goalSize=None,
                 maxEpisodes=None, taskDef=None):

        if initVel is None:
            initVel = [0, 0]
        if initPos is None:
            initPos = [0, 0]
        if goalSize is None:
            goalSize = 4
        if maxEpisodes is None:
            maxEpisodes = 10000
        if taskDef is None:
            taskDef = 'multi'
        self._maxEpisodes = maxEpisodes
        self._n = n
        self.viewer = None
        self._task = None
        self._initPos = initPos  # added by alex
        self._initVel = initVel  # added by alex
        self._goalPos = goalPos  # added by alex
        self._goalSize = goalSize  # added by alex
        self._limUpPos = np.ones(self._n) * self.MAX_POS
        self._limUpVel = np.ones(self._n) * self.MAX_VEL
        self._limUpAcc = np.ones(self._n) * self.MAX_ACC
        self._limUpFor = np.ones(self._n) * self.MAX_FOR
        self.setSpaces()
        #self.enrichObsSpace()
        self.sampleGoal()
        self.state = None
        self.seed()
        self._dt = dt
        self._render = render
        self._t = 0

    @abstractmethod
    def setSpaces(self):
        pass

    def enrichObsSpace(self):
        # enriches observation space by goal, objects, and other relevant information
        # converts observation space to dictionary format
        readObs = self.observation_space
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-self._limUpPos, self._limUpPos, dtype=np.float32),
            task=spaces.Discrete(3), # check for meaningful task identifiers
            observation=readObs
        ))
        return

    def sampleGoal(self):
        #goal = self.observation_space.spaces['desired_goal'].sample()
        goal = spaces.Box(-self._limUpPos, self._limUpPos, dtype=np.float32).sample()
        return goal

    def dt(self):
        return self._dt

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, pos=None, vel=None):
        if not isinstance(pos, np.ndarray) or not pos.size == self._n:
            pos = np.zeros(self._n)
        if not isinstance(vel, np.ndarray) or not vel.size == self._n:
            vel = np.zeros(self._n)
        self.state = np.concatenate((pos, vel))
        self._goalPos = self.sampleGoal()
        self._t = 0
        return self._get_ob()

    def step(self, a):
        s = self.state  # see if this should include obstacles and goals
        self.action = a
        t = self._t
        self._t = t+1
        _ = self.continuous_dynamics(self.state, 0)
        ns = self.integrate()
        self.state = ns
        terminal = self._terminal()
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

        goal_dist = self._goalPos - self.state[0:2] # relative distance to the goal

        # set irrelevant distances to zero for 1D goal spaces
        if self._task == 1:
            goal_dist[1] = 0
        if self._task == 2:
            goal_dist[0] = 0

        obs = np.concatenate([self.state, goal_dist]).flatten()
        return obs

    def _terminal(self):
        s = self.state
        # changed by ALEX
        s = self.state
        is_out_of_window = bool(abs(s[0]) > self.MAX_POS or abs(s[1]) > self.MAX_POS or s[1] == 3)
        goal_pos_reached = point_inside_circle(s[0], s[1], self._goalPos[0], self._goalPos[1], self._goalSize)
        max_episodes_reached = bool(self._t>=self._maxEpisodes)
        # add max episodes
        if is_out_of_window or goal_pos_reached or max_episodes_reached:
            return True
        return False

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def integrate(self):
        x0 = self.state[0 : 2 * self._n]
        t = np.arange(0, 2 * self._dt, self._dt)
        ynext = odeint(self.continuous_dynamics, x0, t)
        return ynext[1]

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        s = self.state
        if s is None:
            return None

        bound = self.MAX_POS
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
