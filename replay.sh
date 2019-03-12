
export GYM_JSBSIM_RENDER_MODE=flightgear

cd baselines
python -m baselines.run --alg=ddpg \
       --env=JSBSim-TurnHeadingControlTask-Cessna172P-Shaping.EXTRA_SEQUENTIAL-NoFG-v0 \
       --num_timesteps=0 \
       --play \
       --play_env=JSBSim-TurnHeadingControlTask-Cessna172P-Shaping.EXTRA_SEQUENTIAL-FG-v0
