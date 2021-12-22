import numpy as np
from gym import spaces

from pointRobot.envs.pointRobotEnv import PointRobotEnv


class PointRobotAccEnv(PointRobotEnv):
    def setSpaces(self):
        o = np.concatenate((self._limUpPos, self._limUpVel, self._limUpPos)) ## DELETE last entry only for testing !!
        self.observation_space = spaces.Box(low=-o, high=o, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self._limUpAcc, high=self._limUpAcc, dtype=np.float32
        )

    def continuous_dynamics(self, x, t):
        vel = x[self._n : self._n * 2]
        xdot = np.concatenate((vel, self.action))
        return xdot
