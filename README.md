# jsbsim_rl
reinforcement learning with jsbsim and open ai baselines

Repo not currently cleaned up and doing heavy development on this right now.  Sharing it to collaborate with others.

I'm using both openai baselines and stable-baselines (a fork of openai baselines), trying to
decide which I like best.  stable-baselines has a number of software engineering-y improvements
and consistency across modules that make it potentially a better place to start from.  It's
also very actively developed (as of Nov 2018) and responsive compared to the openai github
repo.

In the root dir, there are train.sh / replay.sh which are meant to use openai baselines code.
However, openai baselines doesn't implement save/load for DDPG, so it doesn't actually work.

train.py and eval.py are the corresponding scripts for stable-baselines, which does implement
save / load for all algorithms.  It has some minor bugs related to checkpointing, but
I'm working them with the maintainers.

For now, I'm pulling in other repos into this repo because I'm freely making changes to them.
At some point I may make pull requests for my changes in the original repos and drop the
copy in this repo, but I'll probably do that once things are stable.

I'm building this repo into a docker image and push it as jrjbertram/jsbsim_rl.  It's currently
and 11GB image, but this is due to all of the software packages it pulls in such as flight gear.

Currently I have gym-jsbsim integrated with openai baselines and I'm able to run training against
jsbsim.  From the tensorboard plots, it looks like it is training successfully.  I'm working on
getting trained model to fly against flightgear (with jsbsim as the underlying sim) within
the image to validate that the trained model flies the airplane well.
