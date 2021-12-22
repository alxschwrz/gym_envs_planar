import numpy as np
from gym import spaces
'''
open question:
    - how to add the task specification?
    ---- as a 'classifier' feature input. Observation_space contains "task_id" as spaces.Discrete()
    ---- implicitly through the reward function. Observation goal_dist for x-dimension is always 0.

# best shot so far:
- simply set the dist_to goal component wise to zero for the different tasks and provide 
  a task identifier 

'''

'''
Task Definitions:
- xy: The goal is to move to a randomly sampled [x,y] position. TaskId: 0
- x:  The goal is to move to a randomly sampled [x]   position. TaskId: 1
- y:  The goal is to move to a randomly sampled [y]   position. TaskId: 2
- multi: random sampling of 'x', 'y', 'xy' tasks
'''
def sampleTask(self):
    if self._taskDef == 'multi':
        self._task = np.random.choice([0, 1, 2])

    if self._taskDef == 'xy':
        self._task = 0
        self.sample2DGoal()
    elif self._taskDef == 'x':
        self._task = 1
        self.sample1DGoal()
    elif self._taskDef == 'y':
        self._task = 2
        self.sample1DGoal()

'''
1-Dimensional Task setting
using zero-padding so far
'''
def sample1DGoalX(self):
    goal = spaces.Box(-self._limUpPos[0], self._limUpPos[0], dtype=np.float32).sample()
    return [goal, None]
def sample1DGoalY(self):
    goal = spaces.Box(-self._limUpPos[0], self._limUpPos[0], dtype=np.float32).sample()
    return [None, goal]
'''
2-Dimensional Task setting
'''
def sample2DGoal(self):
    goal = spaces.Box(-self._limUpPos, self._limUpPos, dtype=np.float32).sample()
    return goal

def calculate_reward(self):
    reward = 0
    s = self.state

    ##PSEUDO
    # if goal_dist < thres:
    #    return 100

    if point_inside_circle(s[0], s[1], self._goalPos[0], self._goalPos[1], self._goalSize):
        # goal reached
        return 100
    if bool(abs(s[0]) > self.MAX_POS or abs(s[1]) > self.MAX_POS or s[1] == 3):
        # out of window
        return -10
    return -0.1


