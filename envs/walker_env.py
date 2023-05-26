from matplotlib import pyplot as plt
import numpy as np
import math
from scipy import interpolate
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, root
from dataclasses import dataclass, field 
from typing import List


@dataclass
class Parameters:
    leg_length: float   = 1
    gravity: float  = 1
    leg_mass: float = 0.5
    hip_mass: float = 1
    leg_inertia: float = 0.02
    com_dist   : float = 0.5   # from the hip
    slope      : float = 0.01    


class DataLogger:
    def __init__(self):
        self.data = {}
        self.fig = None
        self.ax = None
        self.fps = 10

    def log_data(self,**kwargs):
        for parameter, value in kwargs.items():
            if parameter not in self.data:
                self.data[parameter] = []
            self.data[parameter].append(value)
        
    def plot_data(self):
        times = self.data['time']
        states = self.data['states']
        hip_coordinates = self.data['hip_coordinates']
        th_stance    = [state[0] for state in states]
        omega_stance = [state[1] for state in states]
        th_stance_swing = [state[2] for state in states]
        omega_stance_swing = [state[3] for state in states]
        collision_val    = [state[2]+2*state[0] for state in states]
        xh =[hip_coordinate[0] for hip_coordinate in hip_coordinates]
        yh =[hip_coordinate[1] for hip_coordinate in hip_coordinates]
        
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(times,th_stance,'k--')
        plt.plot(times,th_stance_swing,'r')
        plt.ylabel('theta')
        plt.subplot(2,1,2)
        plt.plot(times,omega_stance,'k--')
        plt.plot(times,omega_stance_swing,'r')
        plt.ylabel('thetadot')
        plt.xlabel('time')
        
        plt.figure()
        plt.plot(times,collision_val,'k')

        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(times,xh,'b')
        plt.ylabel('xh')
        plt.subplot(2,1,2)
        plt.plot(times,yh,'b')
        plt.ylabel('yh')
        plt.xlabel('time')
        plt.show()
    
    def playback_data(self):
        times = self.data['time']
        states = self.data['states']
        # Iterate over the lists simultaneously and print each row
        for i in range(len(times)):
            time = times[i]
            state = states[i]
            print(f"Time: {time}, State: {state}")
    
    def animate_data(self):
        if self.fig is None and self.ax is None:
            # Create the figure and axes for the initial render
            self.fig, self.ax = plt.subplots()
            self.ax.set_aspect('equal')
            xmin = -1 
            xmax = 10
            ymin = -0.1
            ymax = 2
            self.ax.set_xlim(-1, 10)  # Adjust the limits as needed
            self.ax.set_ylim(-0.1, 2)  # Adjust the limits as needed
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_title('2D Walker')
            ramp, = self.ax.plot([xmin, xmax],[0,0],linewidth=1, color='blue')

        times                           = self.data['time']
        states                          = np.stack(self.data['states'])
        hip_coordinates                 = np.stack(self.data['hip_coordinates'])
        stance_foot_coordinates         = np.stack(self.data['stance_foot_coordinates'])
        swing_foot_coordinates          = np.stack(self.data['swing_foot_coordinates'])   
        z = np.concatenate((states,stance_foot_coordinates,hip_coordinates, swing_foot_coordinates), axis=1)
        t =[time for time in times]
        data_pts = 1/self.fps
        t_interp = np.arange(t[0],t[len(t)-1],data_pts)
        [m,n] = np.shape(z)
        shape = (len(t_interp),n)
        z_interp = np.zeros(shape)
        for i in range(0,n):
            f = interpolate.interp1d(t, z[:,i])
            z_interp[:,i] = f(t_interp)
        for i in range(0,len(t_interp)):
            C1     = z_interp[i,4:6]
            H      = z_interp[i,6:8]
            C2     = z_interp[i,8:10]
            hip, = self.ax.plot(H[0],H[1],color='black',marker='o',markersize=10)
            leg1, = self.ax.plot([H[0], C1[0]],[H[1], C1[1]],linewidth=5, color='green')
            leg2, = self.ax.plot([H[0], C2[0]],[H[1], C2[1]],linewidth=5, color='green')
            plt.pause(0.01)
            
            if (i<len(t_interp)-1):
                hip.remove()
                leg1.remove()
                leg2.remove()
        plt.show()
    
class CompassGaitWalker:
    def __init__(self,p = Parameters()):
        self.params = p
        self._stance_foot_coordinates = [0,0]
        self.log_enable = 0
        self.fps = 5
        self.fig = None
        self.ax = None
        self.logger = DataLogger()
        self.dt = 0.001
        self.collision_event = False
        self.gstop_value = 0
        self.prev_gstop_value = 0

    def step(self,t,z,u):
        # one step of integration  for dt time for given control u(hip torque, pushoff force)
        if self.log_enable:
            data = {
                    'time': t,
                    'states': z,
                    'hip_coordinates': self._get_hip_coordinates(z[0],z[2]),
                    'stance_foot_coordinates' : self._stance_foot_coordinates,
                    'swing_foot_coordinates'  : self._get_footstrike_coordinates(z[0],z[2])
                    }
            self.logger.log_data(**data)
        znext = np.array([0,0,0,0],dtype=float)
        dz = self.single_stance_dynamics(t,z,u)
        for i in range(len(z)):
            znext[i] = z[i] + dz[i]*self.dt
        self.collision_event = False
        self.gstop_value = self.detect_collision(t,znext)
        if (self.gstop_value != 1) and (detect_zero_crossing(self.prev_gstop_value) != detect_zero_crossing(self.gstop_value)):
            self.collision_event = True
            zminus = znext
            self.z_bfs = zminus 
            self._footstrike_coordinates = self._get_footstrike_coordinates(self.z_bfs[0],self.z_bfs[2])
            zplus = self.impact_map(t,zminus)
            znext = zplus
            self._update_stance_foot_coordinates()
        self.prev_gstop_value = self.gstop_value
        return znext
        
    def single_stance_dynamics(self,t,z,u=[0,0]):
        # implementation of single stance dynamics
        
        th_stance,omega_stance,the_stance_swing,omega_stance_swing = z
        M = self.params.hip_mass
        m = self.params.leg_mass
        I = self.params.leg_inertia
        c = self.params.com_dist
        l = self.params.leg_length
        g = self.params.gravity
        slope = self.params.slope
        
        A11 = 2.0*I + M*l**2 + m*(c - l)**2 + m*(c**2 - 2*c*l*math.cos(the_stance_swing) + l**2)
        A12 = 1.0*I + c*m*(c - l*math.cos(the_stance_swing))
        A21 = 1.0*I + c*m*(c - l*math.cos(the_stance_swing))
        A22 = 1.0*I + c**2*m 
        b1 =  -M*g*l*math.sin(slope - th_stance) \
                + c*g*m*math.sin(slope - th_stance) \
                - c*g*m*math.sin(-slope + th_stance + the_stance_swing) \
                - 2*c*l*m*omega_stance*omega_stance_swing*math.sin(the_stance_swing) - c*l*m*omega_stance_swing**2*math.sin(the_stance_swing) \
                - 2*g*l*m*math.sin(slope - th_stance)
        b2 =  1.0*c*m*(-g*math.sin(-slope + th_stance + the_stance_swing) + l*omega_stance**2*math.sin(the_stance_swing)) + u[0]

        A_ss = np.array([[A11,A12],[A21,A22]])
        b_ss = np.array([b1,b2])
        invA_ss = np.linalg.inv(A_ss)
        thetaddot = invA_ss.dot(b_ss) 
        alpha_stance = thetaddot[0]
        alpha_stance_swing = thetaddot[1]
        return [omega_stance,alpha_stance,omega_stance_swing,alpha_stance_swing]
    
    def impact_map(self,t,z):
        # implementation of collision equation and configuration variables name change
        th_stance_n,omega_stance_n,th_stance_swing_n,omega_stance_swing_n = z

        th_stance = th_stance_n + th_stance_swing_n
        th_stance_swing = -th_stance_swing_n

        M = self.params.hip_mass
        m = self.params.leg_mass
        I = self.params.leg_inertia
        c = self.params.com_dist
        l = self.params.leg_length
        g = self.params.gravity
        slope = self.params.slope

        J11 =  1
        J12 =  0
        J13 =  l*(-math.cos(th_stance_n) + math.cos(th_stance_n + th_stance_swing_n))
        J14 =  l*math.cos(th_stance_n + th_stance_swing_n)
        J21 =  0
        J22 =  1
        J23 =  l*(-math.sin(th_stance_n) + math.sin(th_stance_n + th_stance_swing_n))
        J24 =  l*math.sin(th_stance_n + th_stance_swing_n)

        J = np.array([[J11, J12, J13, J14], [J21,J22,J23,J24]])

        A11 =  1.0*M + 2.0*m
        A12 =  0
        A13 =  -1.0*M*l*math.cos(th_stance_n) + m*(c - l)*math.cos(th_stance_n) + 1.0*m*(c*math.cos(th_stance_n + th_stance_swing_n) - l*math.cos(th_stance_n))
        A14 =  1.0*c*m*math.cos(th_stance_n + th_stance_swing_n)
        A21 =  0
        A22 =  1.0*M + 2.0*m
        A23 =  -1.0*M*l*math.sin(th_stance_n) + m*(c - l)*math.sin(th_stance_n) + m*(c*math.sin(th_stance_n + th_stance_swing_n) - l*math.sin(th_stance_n))
        A24 =  1.0*c*m*math.sin(th_stance_n + th_stance_swing_n)
        A31 =  -1.0*M*l*math.cos(th_stance_n) + m*(c - l)*math.cos(th_stance_n) + 1.0*m*(c*math.cos(th_stance_n + th_stance_swing_n) - l*math.cos(th_stance_n))
        A32 =  -1.0*M*l*math.sin(th_stance_n) + m*(c - l)*math.sin(th_stance_n) + m*(c*math.sin(th_stance_n + th_stance_swing_n) - l*math.sin(th_stance_n))
        A33 =  2.0*I + M*l**2 + m*(c - l)**2 + m*(c**2 - 2*c*l*math.cos(th_stance_swing_n) + l**2)
        A34 =  1.0*I + c*m*(c - l*math.cos(th_stance_swing_n))
        A41 =  1.0*c*m*math.cos(th_stance_n + th_stance_swing_n)
        A42 =  1.0*c*m*math.sin(th_stance_n + th_stance_swing_n)
        A43 =  1.0*I + c*m*(c - l*math.cos(th_stance_swing_n))
        A44 =  1.0*I + c**2*m
        A_n_hs = np.array([[A11, A12, A13, A14], [A21, A22, A23, A24], [A31, A32, A33, A34], [A41, A42, A43, A44]])

        X_n_hs = np.array([0, 0, omega_stance_n, omega_stance_swing_n])
        b_temp  = A_n_hs.dot(X_n_hs)
        b_hs = np.block([ b_temp, 0, 0 ])
        zeros_22 = np.zeros((2,2))
        A_hs = np.block([[A_n_hs, -np.transpose(J)] , [ J, zeros_22] ])
        invA_hs = np.linalg.inv(A_hs)
        X_hs = invA_hs.dot(b_hs)
        omega_stance = X_hs[2] + X_hs[3]
        omega_stance_swing = -X_hs[3]

        return [th_stance,omega_stance,th_stance_swing,omega_stance_swing]

    def detect_collision(self,t,z):
        
        th_stance,omega_stance,th_stance_swing,omega_stance_swing = z
        #gstop = th_stance_swing + 2*th_stance
        if (th_stance > -0.05):
            gstop = 1
        else:
            gstop = th_stance_swing + 2*th_stance
        return gstop      
        
    def _get_hip_coordinates(self,th_stance,th_stance_swing):
        l = self.params.leg_length
        return np.array([self._stance_foot_coordinates[0]-l*math.sin(th_stance), self._stance_foot_coordinates[1] + l*math.cos(th_stance)])
    
    def _get_footstrike_coordinates(self,th_stance,th_stance_swing):
        l = self.params.leg_length
        x_swing = l*(math.sin(th_stance)*math.cos(th_stance_swing) + math.sin(th_stance_swing)*math.cos(th_stance)) - l*math.sin(th_stance) + self._stance_foot_coordinates[0]
        y_swing = l*(math.sin(th_stance)*math.sin(th_stance_swing) - math.cos(th_stance)*math.cos(th_stance_swing)) + l*math.cos(th_stance) + self._stance_foot_coordinates[1]
        return np.array([x_swing,y_swing])
    
    def _update_stance_foot_coordinates(self):
        self._stance_foot_coordinates = self._footstrike_coordinates
    
    def render(self):
        pass
    
    def reset(self):
        pass
    
class SimplestWalker:
    def __init__(self,p = Parameters()): # only needs slope really !!
        self.params = p
        self._stance_foot_coordinates = [0,0]
        self.log_enable = 0
        self.fps = 10
        self.fig = None
        self.ax = None
        self.logger = DataLogger()
        self.dt = 0.001
        self.collision_event = False
        self.gstop_value = 0
        self.prev_gstop_value = 0
        
    def step(self,t,z,u):
        # one step of integration  for dt time for given control u(hip torque, pushoff force)
        if self.log_enable:
            data = {
                    'time': t,
                    'states': z,
                    'hip_coordinates': self._get_hip_coordinates(z[0],z[2]),
                    'stance_foot_coordinates' : self._stance_foot_coordinates,
                    'swing_foot_coordinates'  : self._get_footstrike_coordinates(z[0],z[2])
                    }
            self.logger.log_data(**data)
        znext = np.array([0,0,0,0],dtype=float)
        dz = self.single_stance_dynamics(t,z,u)
        for i in range(len(z)):
            znext[i] = z[i] + dz[i]*self.dt
        self.collision_event = False
        self.gstop_value = self.detect_collision(t,znext)
        if (self.gstop_value != 1) and (detect_zero_crossing(self.prev_gstop_value) != detect_zero_crossing(self.gstop_value)):        
            self.collision_event = True
            zminus = znext
            self.z_bfs = zminus 
            self._footstrike_coordinates = self._get_footstrike_coordinates(self.z_bfs[0],self.z_bfs[2])
            zplus = self.impact_map(t,zminus)
            znext = zplus
            self._update_stance_foot_coordinates()
        self.prev_gstop_value = self.gstop_value
        return znext
    def single_stance_dynamics(self,t,z,u=[0,0]):
        # implementation of single stance dynamics
        
        the,thedot,phi,phidot = z
        slope = self.params.slope
        
        theddot = math.sin(the-slope)
        phiddot = -math.sin(the-slope) + (thedot**2 - math.cos(the-slope))*math.sin(phi) + u[0]
        return [thedot,theddot,phidot,phiddot]
    
    def impact_map(self,t,z):
        # implementation of collision equation and configuration variables name change
        the,thedot,phi,phidot = z
        the_plus    = -the
        thedot_plus = math.cos(2*the)*thedot
        phi_plus    = 2*the
        phidot_plus = -(1-math.cos(2*the))*thedot_plus
        return [the_plus,thedot_plus,phi_plus,phidot_plus]

    def detect_collision(self,t,z):
        
        the,thedot,phi,phidot = z
        #gstop = th_stance_swing + 2*th_stance
        if (the > -0.05):
            gstop = 1
        else:
            gstop = phi + 2*the
        return gstop      
        
    def _get_hip_coordinates(self,th_stance,th_stance_swing):
        l = self.params.leg_length
        return np.array([self._stance_foot_coordinates[0]-l*math.sin(th_stance), self._stance_foot_coordinates[1] + l*math.cos(th_stance)])
    
    def _get_footstrike_coordinates(self,th_stance,th_stance_swing):
        l = self.params.leg_length
        x_swing = l*(math.sin(th_stance)*math.cos(th_stance_swing) + math.sin(th_stance_swing)*math.cos(th_stance)) - l*math.sin(th_stance) + self._stance_foot_coordinates[0]
        y_swing = l*(math.sin(th_stance)*math.sin(th_stance_swing) - math.cos(th_stance)*math.cos(th_stance_swing)) + l*math.cos(th_stance) + self._stance_foot_coordinates[1]
        return np.array([x_swing,y_swing])
    
    def _update_stance_foot_coordinates(self):
        self._stance_foot_coordinates = self._footstrike_coordinates
        
    def render(self):
        pass
    
    def reset(self):
        pass

           
def detect_zero_crossing(x):
    return x > 0

    
class InvertedPendulumWalker():
    def __init__(self):
        pass