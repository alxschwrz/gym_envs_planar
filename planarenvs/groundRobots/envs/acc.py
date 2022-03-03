import numpy as np
from gym import spaces

from planarenvs.groundRobots.envs.groundRobotEnv import GroundRobotEnv


class GroundRobotAccEnv(GroundRobotEnv):
    def setSpaces(self):
        self.observation_space = spaces.Dict({
            'x': spaces.Box(low=-self._limUpPos, high=self._limUpPos, dtype=np.float64), 
            'xdot': spaces.Box(low=-self._limUpVel, high=self._limUpVel, dtype=np.float64), 
            'vel': spaces.Box(low=-self._limUpRelVel, high=self._limUpRelVel, dtype=np.float64), 
        })
        self.action_space = spaces.Box(
            low=-self._limUpRelAcc, high=self._limUpRelAcc, dtype=np.float64
        )

    def integrate(self):
        super().integrate()
        self.state['xdot'] = self.computeXdot(self.state['x'], self.state['vel'])

    def continuous_dynamics(self, x, t):
        # state = [x, y, theta, vel_rel, xdot, ydot, thetadot]
        x_pos = x[0:3]
        vel = x[3:5]
        xdot = self.computeXdot(x_pos, vel)
        veldot = self.action
        xddot = np.zeros(3)
        return np.concatenate((xdot, veldot, xddot))
