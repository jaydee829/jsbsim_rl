import gym
import time
env = gym.make('SpaceInvaders-v0')
env.reset()
env.render()

print( 'Sleeping for 10 seconds before exiting' )
time.sleep(10)

