import gym
import matplotlib
import jsbsim
import gym_jsbsim
import h5py
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines import logger

matplotlib.use('TkAGG')

# Necessary for it to generate tensorboard log files during the run.  Note that
# in the docker file I set OPENAI_LOGDIR and OPENAI_LOG_FORMAT environment variables
# to generate the kind of logs that I want.
logger.configure()

n_cpu = 8
env = gym.make('JSBSim-TurnHeadingControlTask-F15-Shaping.STANDARD-FG-v0')
env = SubprocVecEnv([lambda: env for i in range(n_cpu)])
# env = DummyVecEnv([lambda:env])

model_name = 'ppo2_fg_steadyheading_rnn_h128_lr1e-4_f15'

model = PPO2.load("model/"+model_name, env=env, tensorboard_log="model/tensorboard/")

obs = env.reset()

eval_steps = 10000
obs_ds = '/turnheading/F15/'+model_name+'/obs'
act_ds = '/turnheading/F15/'+model_name+'/act'


with h5py.File('/home/jsbsim/results/state_action_history.h5', 'a') as file:
    file.create_dataset(obs_ds, maxshape=(env.num_envs, env.observation_space.shape[0], None),
                        data=np.resize(obs, (env.num_envs, env.observation_space.shape[0], 1)))
    file.create_dataset(act_ds, maxshape=(env.num_envs, env.action_space.shape[0], None),
                        data=np.empty((env.num_envs, env.action_space.shape[0], 1)))

    i = 0
    while i < eval_steps:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        i += 1
        file[obs_ds].resize(file[obs_ds].shape[2] + 1, axis=2)
        file[obs_ds][..., -1] = obs
        file[act_ds].resize(file[act_ds].shape[2] + 1, axis=2)
        file[act_ds][..., -1] = action
