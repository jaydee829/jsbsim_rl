
export GYM_JSBSIM_RENDER_MODE=human

cd baselines
python -m baselines.run --alg=ddpg \
       --env=JSBSim-TurnHeadingControlTask-Cessna172P-Shaping.STANDARD-FG-v0 \
       --num_timesteps=1e6 \
       --render=True \
       --save_path=/home/jsbsim/models/ddpg
