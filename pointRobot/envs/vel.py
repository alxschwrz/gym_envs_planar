import numpy as np
from gym import spaces

from pointRobot.envs.pointRobotEnv import PointRobotEnv


class PointRobotVelEnv(PointRobotEnv):
    def setSpaces(self):
        o = np.concatenate((self._limUpPos, self._limUpVel, self._limUpPos)) ## DELETE last entry only for testing !!
        self.observation_space = spaces.Box(low=-o, high=o, dtype=np.float64)
        self.action_space = spaces.Box(
            low=-self._limUpVel, high=self._limUpVel, dtype=np.float64
        )

    def continuous_dynamics(self, x, t):
        vel = self.action
        acc = np.zeros(self._n)
        self.state[self._n: 2 * self._n] = vel
        xdot = np.concatenate((vel, acc))
        return xdot
