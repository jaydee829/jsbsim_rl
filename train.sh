
cd baselines
python -m baselines.run --alg=ddpg \
       --env=JSBSim-TurnHeadingControlTask-Cessna172P-Shaping.STANDARD-NoFG-v0 \
       --num_timesteps=1e6 \
       --play \
       --play_env=JSBSim-TurnHeadingControlTask-Cessna172P-Shaping.STANDARD-FG-v0
       #--save_path=/home/jsbsim/models/ddpg
