
cd baselines
python -m baselines.run --alg=ddpg \
       --env=JSBSim-TurnHeadingControlTask-Cessna172P-Shaping.STANDARD-NoFG-v0 \
       --num_timesteps=0 \
       --load_path=/home/jsbsim/models/ddpg --play
