from walker_env import CompassGaitWalker, Parameters
from matplotlib import pyplot as plt
import numpy as np
import math
from scipy import interpolate
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, root
from dataclasses import dataclass, field 
from typing import List

class PassiveCompassGaitWalker(CompassGaitWalker):
    def __init__(self,p = Parameters()):
        super().__init__(p)
        
    def fixed_point(self,zg):
        zstar = fsolve(self.poincare_map, zg,full_output=True) 
        return zstar[0]

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
            # if (self.render_mode == 1):
            #     self.animate_one_step(t1,z1)
            self._update_stance_foot_coordinates()
            tc = t1[-1]
            zc = z1[-1,:] 
            t_list.extend(t1)
            z_list = np.concatenate((z_list,z1),axis=0) 
        return t_list, z_list
    
    def poincare_map(self,z):
        t = 0
        [t1,z1] = self.one_gait_step(t,z)
        N = len(t1)-1
        return [z1[N,0]-z[0], z1[N,1]-z[1],z1[N,2]-z[2],z1[N,3]-z[3]]
    
    def impact_map_jacobian(self,z):
        pass
    
    def animate_one_step(self,t,z):
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
    
if __name__ == '__main__':
    p = Parameters()
    q1 = 0.2; u1 = -0.25; q2 = -0.4; u2 = 0.2
    z0 = np.array([q1,u1,q2,u2])
    passive_walker = PassiveCompassGaitWalker(p)
    zstar=passive_walker.fixed_point(z0)
    print(zstar)
    passive_walker.log_enable = 1
    passive_walker.logger.fps = 10
    [t,z] = passive_walker.simulate(0,zstar,10)
    passive_walker.logger.plot_data()
    passive_walker.logger.animate_data()
    