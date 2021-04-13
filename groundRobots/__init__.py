from gym.envs.registration import register
register(
    id='ground-robot-vel-v0',
    entry_point='groundRobots.envs:GroundRobotVelEnv'
)
register(
    id='ground-robot-acc-v0',
    entry_point='groundRobots.envs:GroundRobotAccEnv'
)
register(
    id='ground-robot-diffdrive-vel-v0',
    entry_point='groundRobots.envs:GroundRobotDiffDriveVelEnv'
)
register(
    id='ground-robot-diffdrive-acc-v0',
    entry_point='groundRobots.envs:GroundRobotDiffDriveAccEnv'
)