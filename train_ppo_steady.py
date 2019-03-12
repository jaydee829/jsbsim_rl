import gym

import jsbsim
import gym_jsbsim

from stable_baselines.common.policies import LstmPolicy, MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines import logger

# Necessary for it to generate tensorboard log files during the run.  Note that
# in the docker file I set OPENAI_LOGDIR and OPENAI_LOG_FORMAT environment variables
# to generate the kind of logs that I want.
logger.configure()

# multiprocess environment
n_cpu = 8
env = gym.make('JSBSim-HeadingControlTask-F15-Shaping.STANDARD-NoFG-v0')
env = SubprocVecEnv([lambda: env for i in range(n_cpu)])

model_name = 'ppo2_fg_steadyheading_rnn_h128_lr1e-4_f15'

try:
    model = PPO2.load("model/"+model_name, env=env, tensorboard_log="model/tensorboard/")
    model.set_env( env )

except ValueError:  # Model doesn't exist

    model = PPO2(LstmPolicy, env, n_steps=128, verbose=1, nminibatches=4, gamma=0.95, learning_rate=1.0e-4, tensorboard_log="model/tensorboard/", policy_kwargs={'feature_extraction': 'None'})

while True:
    model.learn(total_timesteps=1000000)
    model.save("model/"+model_name)
