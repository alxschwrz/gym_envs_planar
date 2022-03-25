import gym
import planarenvs.pointRobot
import numpy as np
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import check_for_correct_spaces


def main():
    FLAG_VEC_NORM = False

    env = gym.make("point-robot-vel-v0", render=False, dt=0.01)
    check_env(env)
    check_for_correct_spaces(env, env.observation_space, env.action_space)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])


    eval_env = gym.make("point-robot-vel-v0", render=True, dt=0.01)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])

    if FLAG_VEC_NORM:
        env = VecNormalize(env, norm_obs=True, norm_reward=False,
                           clip_obs=10.)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                                clip_obs=10.)

    #model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="tensorboard_test/")
    #model = PPO("MultiInputPolicy", env, verbose=1)

    model = SAC('MultiInputPolicy', env, verbose=1)
    #model = SAC('MultiInputPolicy', env, verbose=2, tensorboard_log="tensorboard_test/")

    print('starting training...')
    model.learn(total_timesteps=1_000_000, log_interval=4, tb_log_name="PPO_Rand")

    #model.save("models/PPO_TESTING1")
    #del model
    #model = PPO.load("models/PPO_TESTING1")
    model = SAC.load("models/SAC_TESTING")

    n_episodes = 20
    n_steps = 1000
    cumReward = 0.0

    for e in range(n_episodes):
        # ob = env.reset(pos=initPos, vel=initVel)
        ob = eval_env.reset()

        print("Starting episode")
        # t = 0
        for i in range(n_steps):
            # t = t + eval_env.dt()
            defaultAction = np.array([0.19, 0.50])
            # action = defaultAction  # (t)
            action, _states = model.predict(ob, deterministic=True)
            ob, reward, done, info = eval_env.step(action)
            if (i % 100) == 1:
                print("it: {}, act: {}, obs: {}, reward: {}".format(i, action, ob, reward))
                # print(ob, reward, done, info)
            cumReward += reward
            if done:
                break


if __name__ == "__main__":
    main()
