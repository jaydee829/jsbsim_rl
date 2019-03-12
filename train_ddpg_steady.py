import gym
import numpy as np

import argparse

import jsbsim
import gym_jsbsim

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines import logger

# Necessary for it to generate tensorboard log files during the run.  Note that
# in the docker file I set OPENAI_LOGDIR and OPENAI_LOG_FORMAT environment variables
# to generate the kind of logs that I want.
logger.configure()

env = gym.make('JSBSim-HeadingControlTask-Cessna172P-Shaping.STANDARD-NoFG-v0')
env = DummyVecEnv([lambda: env])

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

try:
    model = DDPG.load("model/ddpg_fg_steadyheading", env=env, tensorboard_log="model/tensorboard/")
    model.set_env( env )

except ValueError:  # Model doesn't exist

    model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, normalize_observations=True, tensorboard_log="model/tensorboard/")

while True:
    model.learn(total_timesteps=1e6)
    model.save("model/ddpg_fg_steadyheading")
