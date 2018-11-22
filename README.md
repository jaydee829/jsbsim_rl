# jsbsim_rl
Using reinforcement learning to fly an airplane. 

# Introduction

JSBSim is an open source flight dyamics simulator that the open source flight simulator FlightGear uses under the hood.  This project aims to string together a number of software libraries and tools to create an end-to-end reinforcement learning environment for learning to fly airplanes.

# Dependencies

 * JSBSim (optionally with FlightGear for visualization)
 * OpenAI Gym
 * gym-jsbsim, https://github.com/Gor-Ren/gym-jsbsim, an openai gym environment wrapper class for jsbsim and flightgear.
 * Algorithm Implementations:
     * openai baselines
     * stable-baselines

# Implementation Notes:
Repo not currently cleaned up and doing heavy development on this right now.  Sharing it to collaborate with others.

I'm using both openai baselines and stable-baselines (a fork of openai baselines), trying to
decide which I like best.  stable-baselines has a number of software engineering-y improvements
and consistency across modules that make it potentially a better place to start from.  It's
also very actively developed (as of Nov 2018) and responsive compared to the openai github
repo.

In the root dir, there are train.sh / replay.sh which are meant to use openai baselines code.
However, openai baselines doesn't implement save/load for DDPG, so it doesn't actually work.
I have added in checkpointing but I'm not confident it is actually working yet.

train.py and eval.py are the corresponding scripts for stable-baselines, which does implement
save / load for all algorithms.  It has some minor bugs related to checkpointing, but
I'm working them with the maintainers.

For now, I'm pulling in other repos into this repo because I'm freely making changes to them.
At some point I may make pull requests for my changes in the original repos and drop the
copy in this repo, but I'll probably do that once things are stable.

Currently I have gym-jsbsim integrated with openai baselines and I'm able to run training against
jsbsim.  From the tensorboard plots, it looks like it is training successfully.  I'm working on
getting trained model to fly against flightgear (with jsbsim as the underlying sim) within
the image to validate that the trained model flies the airplane well.

# Usage

Need to document this once it becomes more stable.

# Docker Images

I'm building this repo into a docker image and push it as jrjbertram/jsbsim_rl.  It's currently
and 11GB image, but this is due to all of the software packages it pulls in such as flight gear.

The image contains all the dependencies needed to perform deep learning within an nvidia-docker 
container.  It also has a VNC server (and a web-based noVNC server) embedded within it so that
the user can log into a desktop that runs within the container to view visualization.

The image also pulls in all the code from this repo and builds all the dependencies related to 
jsbsim.  

I run this image on various servers which have nvidia-docker installed on them.  The image has 
full access to the GPU.  Note that the nvidia-docker host must be linux... nvidia-docker doesn't
work on windows.  I typically use docker-machine from my laptop to launch the image on the 
(sometimes remote or cloud based) servers, and then log into the desktop from my laptop's web
browser.  I kick off the runs from terminal windows within the browser-based VNC session and
then let them run overnight or for extended runs, and reconnect the browser-based VNC session
periodically to check on the progress of the runs.  I typically also create and mount docker
volumes to preserve the logs and saved models which allows me to restart the docker containers
without losing progress on training.

