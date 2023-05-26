import sys
import os
parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(parent_folder_path)
import numpy as np
import math
from envs.walker_env import Parameters, SimplestWalker
from simplest_walker_control import simplest_walker_passive_fpc

# instantiate model and parameters
p = Parameters(1,10,0.05,2,0,1,0.1155)    # actually only slope required for simplest walker
walker = SimplestWalker(p)
control = simplest_walker_passive_fpc(p) 
control.swing_leg_velocity_at_footstrike = -.1  # free parameter dictates stability
walker.dt = 0.001
control.dt = 0.001
walker.log_enable = 1
theta_des = 0.15   # desired foot placement
N = 10            # number of steps to walk

# get steady state gait cycle parameters
z0 = control.compute_state_for_steady_gait_cycle(stance_foot_angle=theta_des) # similar to passive walker, with this z0, we can achieve periodic walking
tstep = control.compute_tstep() # time of one step

# simulate for N steps
total_time = N*tstep
num_steps = (int)(total_time/walker.dt)
curr_state = z0
for step in range(num_steps):
    time = step*walker.dt
    if step % (walker.dt/control.dt) == 0:
        u = control.compute_open_loop_control(walker.collision_event,curr_state)
    next_state = walker.step(time,curr_state,u)
    curr_state= next_state

walker.logger.plot_data()
walker.logger.animate_data()



 


