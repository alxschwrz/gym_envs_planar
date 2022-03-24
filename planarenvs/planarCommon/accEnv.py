import numpy as np
from gym import spaces


class AccEnv(object):
    def setSpaces(self):
        self.observation_space = spaces.Dict({
            'x': spaces.Box(low=-self._limUpPos, high=self._limUpPos, dtype=np.float32),
            'xdot': spaces.Box(low=-self._limUpVel, high=self._limUpVel, dtype=np.float32)
        })
        self.action_space = spaces.Box(
            low=-self._limUpAcc, high=self._limUpAcc, dtype=np.float32
        )

    def continuous_dynamics(self, x, t):
        vel = x[self._n : self._n * 2]
        xdot = np.concatenate((vel, self.action))
        return xdot
