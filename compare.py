import gym
import numpy as np
import os

import argparse
from stable_baselines.common.misc_util import boolean_flag


def parse_args():
    """
    parse the arguments for DDPG training

    :return: (dict) the arguments
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--baseline', required=False, choices=['openai', 'stable'], default='stable' )
    parser.add_argument('--env',      required=False, choices=['lunar', 'jsbsim'], default='lunar' )
    parser.add_argument('--algo',     required=False, choices=['ddpg', 'ppo2'], default='ddpg' )
    parser.add_argument('--mode',                     choices=['train', 'play'], default='train' )
    parser.add_argument('--steps',    type=float,  default=1e6)
    parser.add_argument('--prefix',                default=None )

    args = parser.parse_args()
    return args


args = parse_args()

basedir = "model/%s/%s/%s" % (args.baseline, args.algo, args.env)

if args.prefix:
    basedir = "%s/%s" % (basedir, args.prefix)

try:
    os.makedirs(basedir)
    print("Directory " , basedir ,  " Created ")
except FileExistsError:
    pass

os.environ[ 'OPENAI_LOGDIR' ] = basedir
os.environ[ 'OPENAI_LOG_FORMAT' ] = 'stdout,tensorboard'

reload_model = None

if args.baseline == 'stable':
    from stable_baselines.common.vec_env import DummyVecEnv

    # Necessary for it to generate tensorboard log files during the run.  Note that
    # in the docker file I set OPENAI_LOGDIR and OPENAI_LOG_FORMAT environment variables
    # to generate the kind of logs that I want.
    from stable_baselines import logger
    print( 'Configuring stable-baselines logger')
    logger.configure()

    if args.env == 'jsbsim':
        import jsbsim
        import gym_jsbsim

        env = gym.make('JSBSim-TurnHeadingControlTask-Cessna172P-Shaping.STANDARD-FG-v0')
        env = DummyVecEnv([lambda: env])
    elif args.env == 'lunar':
        env = gym.make('LunarLanderContinuous-v2')
        env = DummyVecEnv([lambda: env])

    if args.algo == 'ddpg':
        from stable_baselines.ddpg.policies import MlpPolicy
        from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
        from stable_baselines import DDPG

        # the noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        param_noise = None
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

        def reload_ddpg_model():
            return DDPG.load('%s/model.ckpt' % basedir, env=env, tensorboard_log="%s/tb/" % basedir, verbose=1)

        reload_model = reload_ddpg_model

        try:
            model = reload_model()
            print( "Loaded model from file")
        except ValueError:  # Model doesn't exist
            model = DDPG(MlpPolicy, env, param_noise=param_noise, action_noise=action_noise, normalize_observations=True, tensorboard_log="%s/tb/" % basedir, verbose=1)
    if args.algo == 'ppo2':
        from stable_baselines import PPO2

        def reload_ppo2_model():
            return PPO2.load('%s/model.ckpt' % basedir, env=env, tensorboard_log="%s/tb/" % basedir, verbose=1)

        reload_model = reload_ppo2_model

        # the noise objects for DDPG
        try:
            model = reload_model()
            print( "Loaded model from file")
        except ValueError:  # Model doesn't exist
            from stable_baselines.common.policies import MlpPolicy
            model = PPO2(MlpPolicy, env, tensorboard_log="%s/tb/" % basedir, verbose=1)
            print( "Created new model from scratch")

else:  # openai baselines
    import sys
    sys.path.insert(0,'baselines')

    from baselines import run
    from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env

    # oa here means openai
    oa_arg_parser = common_arg_parser()
    oa_args, oa_unknown_args = oa_arg_parser.parse_known_args()
    oa_extra_args = {}

    from baselines import logger
    logger.configure()

    if args.env == 'jsbsim':
        import jsbsim
        import gym_jsbsim

        env_id = 'JSBSim-TurnHeadingControlTask-Cessna172P-Shaping.STANDARD-FG-v0'
        #env = gym.make( env_id )
        #from stable_baselines.common.vec_env import DummyVecEnv
        #env = DummyVecEnv([lambda: env])

        env = run.build_env(env_id, oa_args)

    elif args.env == 'lunar':
        env_id = 'LunarLanderContinuous-v2'
        env = run.build_env(env_id, oa_args)
        #env = gym.make(env_id)

    oa_args.env = env_id
    oa_args.play_env = env_id
    oa_args.alg = args.algo



if args.baseline == 'stable':
    if args.mode == 'train':
        while True:
            model.learn(total_timesteps=int(args.steps))
            print( "Saving checkpoint" )
            model.save('%s/model.ckpt' % basedir)
            model = reload_model()
    else:
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()

            done = dones.any() if isinstance( dones, np.ndarray) else dones
            if done:
                obs = env.reset()

else: # openai

    if args.mode == 'train':
        while True:
            model = run.train(oa_args, oa_extra_args, env_id=env_id, env=env)
            # openai doens't implement the save function consistently
            #model.save('%s/model.ckpt' % basedir)
    else: # play
        obs = env.reset()
        def initialize_placeholders(nlstm=128,**kwargs):
            return np.zeros((oa_args.num_env or 1, 2*nlstm)), np.zeros((1))
        state, dones = initialize_placeholders(**oa_extra_args)
        while True:
            actions, _, state, _ = model.step(obs,S=state, M=dones)
            obs, _, done, _ = env.step(actions)
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done

            if done:
                obs = env.reset()
