import numpy as np
import math
from dataclasses import dataclass, field 
from typing import List
from scipy import integrate


class simplest_walker_passive_fpc: 
    """foot placement planning and control for simplest walker model on a slope 
    """
    def __init__(self,model_params):
        self.model_params = model_params
        self.stance_foot_angle = 0
        self.stance_foot_velocity = 0
        self.initial_stance_foot_angle = 0
        self.initial_stance_foot_velocity = 0
        self.control_init = True
        self.swing_leg_velocity_footstrike = 0
        self.tstep = 0
        self.t = 0
        self.dt = 0.001
        self.first = True
        self._poly_coeff = np.zeros([0,0,0,0])

    def set_swing_leg_velocity_at_footstrike(self,phidot):
        self.swing_leg_velocity_footstrike = phidot
        
    def compute_state_for_steady_gait_cycle(self,stance_foot_angle=None,stance_foot_velocity=None):
        slope = self.model_params.slope
        if stance_foot_angle is not None and stance_foot_velocity is None:
            # compute stance foot velocity
            self.initial_stance_foot_angle = stance_foot_angle
            num  = math.sqrt(2.0*(math.cos(stance_foot_angle - slope) - math.cos(stance_foot_angle + slope)))
            den = math.tan(2.0*stance_foot_angle)
            stance_foot_velocity = -num/den
            self.initial_stance_foot_velocity = stance_foot_velocity
        elif stance_foot_angle is None and stance_foot_velocity is not None:
            # compute stance_foot_angle using fsolve/root
            pass
        else:
            ValueError("Invalid Inputs")
        return np.array([self.initial_stance_foot_angle,self.initial_stance_foot_velocity,-2*self.initial_stance_foot_angle,-(1-math.cos(2.0*self.initial_stance_foot_angle))*self.initial_stance_foot_velocity])
            
    def compute_open_loop_control(self,init,state):
        if (self.first or init):
            self.first = False
            self.stance_foot_angle      = self.initial_stance_foot_angle
            self.stance_foot_velocity   = self.initial_stance_foot_velocity
            self.tstep                  = self.compute_tstep()
            self.t                      = 0
            t = self.tstep
            A = np.array([[1,0,0,0],[1, t, t**2, t**3],[0,1,0,0],[0,1,2*t,3*t**2]])
            theta0 = self.stance_foot_angle
            thetadot0 = self.stance_foot_velocity
            b = np.array([-2*theta0,2*theta0,-(1-math.cos(2*theta0))*thetadot0,-self.swing_leg_velocity_footstrike])
            invA = np.linalg.inv(A)
            self._poly_coeff = invA.dot(b)
        # propagate stance dynamics    
        slope = self.model_params.slope
        hip_torque = self.open_loop_swing_leg_polynomial_guidance()
        self.stance_foot_velocity = self.stance_foot_velocity + math.sin(self.stance_foot_angle - slope)*self.dt
        self.stance_foot_angle    = self.stance_foot_angle    + self.stance_foot_velocity*self.dt
        self.t = self.t + self.dt
        return [hip_torque,0] # no push off control in this controller
    
    def compute_closed_loop_control(self,init,state):
        pass
        # if (self.first or init):
        #     self.first = False
        #     #self.initial_stance_foot_angle      = state[0]
        #     #self.initial_stance_foot_velocity   = state[2]
        #     self.tstep                  = self.compute_tstep()
        #     self.t                      = self.tstep
        # hip_torque = self.closed_loop_swing_leg_polynomial_guidance(state)
        # self.t = self.t - self.dt
        # return [hip_torque,0] # no push off control in this controller
    
    def open_loop_swing_leg_polynomial_guidance(self):
        slope = self.model_params.slope
        a0 = self._poly_coeff[0]
        a1 = self._poly_coeff[1]
        a2 = self._poly_coeff[2]
        a3 = self._poly_coeff[3] 
        phi = a3*self.t**3 + a2*self.t**2 + a1*self.t + a0
        phiddot = 2.0*a2 + 6.0*a3*self.t
        return phiddot + math.sin(self.stance_foot_angle - slope) - (self.stance_foot_velocity**2 - math.cos(self.stance_foot_angle - slope))*math.sin(phi)
    
    def closed_loop_swing_leg_polynomial_guidance(self,state):
        pass
        # slope = self.model_params.slope
        # theta    = state[0]
        # thetadot = state[1]
        # phi      = state[2]
        # phidot   = state[3]
        # t = self.t
        # A = np.array([[1,0,0,0],[1, t, t**2, t**3],[0,1,0,0],[0,1,2*t,3*t**2]])
        # theta0 = self.stance_foot_angle
        # thetadot0 = self.stance_foot_velocity
        # b = np.array([phi,-2*self.initial_stance_foot_angle,phidot,-self.swing_leg_velocity_footstrike])
        # invA = np.linalg.inv(A)
        # if (t>=0.5):
        #     self._poly_coeff = invA.dot(b)
        # a0 = self._poly_coeff[0]
        # a1 = self._poly_coeff[1]
        # a2 = self._poly_coeff[2]
        # a3 = self._poly_coeff[3] 
        # phi = a3*self.t**3 + a2*self.t**2 + a1*self.t + a0
        # phiddot = 2.0*a2 + 6.0*a3*self.t
        # return phiddot + math.sin(theta - slope) - (thetadot**2 - math.cos(self.stance_foot_angle - slope))*math.sin(phi)
    
    
    def compute_tstep(self):
        slope = self.model_params.slope
        f = lambda x : 1/math.sqrt(self.initial_stance_foot_velocity**2 + 2*(math.cos(self.initial_stance_foot_angle-slope) - math.cos(x - slope)))
        tstep,error = integrate.quad(f, -self.initial_stance_foot_angle,self.initial_stance_foot_angle)
        return abs(tstep)
    