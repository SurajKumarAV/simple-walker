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
        xh =[hip_coordinate[0] for hip_coordinate in hip_coordinates]
        yh =[hip_coordinate[1] for hip_coordinate in hip_coordinates]
        
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(times,th_stance,'r--')
        plt.plot(times,th_stance_swing,'b')
        plt.ylabel('theta')
        plt.subplot(2,1,2)
        plt.plot(times,omega_stance,'r--')
        plt.plot(times,omega_stance_swing,'b')
        plt.ylabel('thetadot')
        plt.xlabel('time')

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
    
    def animate_data(self,model):
        pass
    
class CompassGaitModel:
    def __init__(self,p = Parameters()):
        self.params = p
        self._stance_foot_coordinates = [0,0]
        self.ankle_pushoff = 0  
        self.hip_torque = 0  
        self.render_mode = 1
        self.log_enable = 0
        self.fps = 10
        self.fig = None
        self.ax = None
        self.logger = DataLogger()
        
    def one_gait_step(self,t0,z0):
        # integrates the equation of motion from one footstrike to another footstrike - one complete gait cycle
        tf = t0+4
        t = np.linspace(t0, tf, 1001)
        event_fun = lambda t, y: self.detect_collision(t, y)
        event_fun.terminal = True
        sol = solve_ivp(self.single_stance_dynamics,[t0, tf],z0,method='RK45', t_eval=t, dense_output=True, \
                    events=event_fun, atol = 1e-13,rtol = 1e-12)

        [m,n] = np.shape(sol.y)
        shape = (n,m)
        t = sol.t
        z = np.zeros(shape)

        [mm,nn,pp] = np.shape(sol.y_events)
        tt_last_event = sol.t_events[mm-1]
        yy_last_event = sol.y_events[mm-1]

        for i in range(0,m):
            z[:,i] = sol.y[i,:]

        t_end = tt_last_event[0]
        theta1 = yy_last_event[0,0]
        omega1 = yy_last_event[0,1]
        theta2 = yy_last_event[0,2]
        omega2 = yy_last_event[0,3]

        zminus = np.array([theta1, omega1, theta2, omega2 ])

        self.z_bfs = zminus 
        self._footstrike_coordinates = self._get_footstrike_coordinates(self.z_bfs[0],self.z_bfs[2])
        zplus = self.impact_map(t_end,zminus)

        t[n-1] = t_end
        z[n-1,0] = zplus[0];
        z[n-1,1] = zplus[1];
        z[n-1,2] = zplus[2];
        z[n-1,3] = zplus[3];

        # if (self.render_mode == 1):
        #     self.render(t,z)
        # self._update_stance_foot_coordinates()
        return t,z
        
    def simulate(self,t0,z0,n_steps=5):
        #call from main to simulate the model
        tc = t0
        zc = z0
        t_list = [tc]
        z_list = np.array(zc).reshape(1,4)
        for i in range(n_steps):
            [t1,z1] = self.one_gait_step(tc,zc)
            if (self.log_enable == 1):
                for i in range(len(t1)-1):
                    
                    data = {
                            'time': t1[i],
                            'states': z1[i],
                            'hip_coordinates': self._get_hip_coordinates(z1[i,0],z1[i,2]),
                            'stance_foot_coordinates' : self._stance_foot_coordinates,
                            'swing_foot_coordinates'  : self._get_footstrike_coordinates(z1[i,0],z1[i,2])
                        }
                    self.logger.log_data(**data)
            if (self.render_mode == 1):
                self.render(t1,z1)
            self._update_stance_foot_coordinates()
            tc = t1[-1]
            zc = z1[-1,:] 
            t_list.extend(t1)
            z_list = np.concatenate((z_list,z1),axis=0) 
        return t_list, z_list  
    
    def single_stance_dynamics(self,t,z):
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
        b2 =  1.0*c*m*(-g*math.sin(-slope + th_stance + the_stance_swing) + l*omega_stance**2*math.sin(the_stance_swing)) 

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
        
        th_stance,omega_stance,th_stance_swing,omega_stance = z
        gstop = th_stance_swing + 2*th_stance
        if (th_stance > -0.05):
            gstop = 1
        else:
            gstop = th_stance_swing + 2*th_stance
        return gstop
    
    def render(self,t,z):
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
            self.ax.set_title('Compass Gait Walker')
            ramp, = self.ax.plot([xmin, xmax],[0,0],linewidth=1, color='blue')

       
        data_pts = 1/self.fps
        t_interp = np.arange(t[0],t[len(t)-1],data_pts)
        [m,n] = np.shape(z)
        shape = (len(t_interp),n)
        z_interp = np.zeros(shape)
        for i in range(0,n):
            f = interpolate.interp1d(t, z[:,i])
            z_interp[:,i] = f(t_interp)
        l = self.params.leg_length
        c = self.params.com_dist
        for i in range(0,len(t_interp)):
            theta1 = z_interp[i,0]
            theta2 = z_interp[i,2]
            C1 = self._stance_foot_coordinates
            H = self._get_hip_coordinates(theta1,theta2)
            C2 = self._get_footstrike_coordinates(theta1,theta2)
            hip, = self.ax.plot(H[0],H[1],color='black',marker='o',markersize=10)
            leg1, = self.ax.plot([H[0], C1[0]],[H[1], C1[1]],linewidth=5, color='green')
            leg2, = self.ax.plot([H[0], C2[0]],[H[1], C2[1]],linewidth=5, color='green')
            plt.pause(0.01)
            
            hip.remove()
            leg1.remove()
            leg2.remove()      
        
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
    
                


    
