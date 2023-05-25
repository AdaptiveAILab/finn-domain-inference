"""
This script provides a class for simulating the shallow water equations
and is based on the work of: https://github.com/jostbr/shallow-water
"""
import matplotlib.pyplot as plt

"""Script that solves the 2D shallow water equations using finite
differences where the momentum equations are taken to be linear, but the
continuity equation is solved in its nonlinear form. The model supports turning
on/off various terms, but in its most complete form, the model solves the following
set of eqations:


    du/dt - fv = -g*d(eta)/dx + tau_x/(rho_0*H)- kappa*u
    
    dv/dt + fu = -g*d(eta)/dy + tau_y/(rho_0*H)- kappa*v
    
    d(eta)/dt + d((eta + H)*u)/dx + d((eta + H)*u)/dy = sigma - w
    
    
where f = f_0 + beta*y can be the full latitude varying coriolis parameter.
For the momentum equations, an ordinary forward-in-time centered-in-space
scheme is used. However, the coriolis terms is not so trivial, and thus, one
first finds a predictor for u, v and then a corrected value is computed in
order to include the coriolis terms. In the continuity equation, it's used a
forward difference for the time derivative and an upwind scheme for the non-
linear terms. The model is stable under the CFL condition of

    dt <= min(dx, dy)/sqrt(g*H)    and    alpha << 1 (if coriolis is used)
    
where dx, dy is the grid spacing in the x- and y-direction respectively, g is
the acceleration of gravity and H is the resting depth of the fluid."""
import numpy as np
import sys
import time


class Simulator:

    def __init__(self, timesteps, width, height, sample_interval = 1, area_scaling = 1, quantity_scaling = 1):
        """
        Constructor method initializing the parameters for the wave equation.
        :param timesteps: the number of timesteps the simulation will be split into
        :param width: Width of the data field
        :param height: Height of the data field
        :param sample_interval: interval between time frames taken from the simulation
        :param area_scaling: increase / decerase the simulated area about the given factor
        :param quantity_scaling: increase / decerase the simulated quantity about the given factor
        :return: No return value
        """
        
        # Set the class parameters
        self.timesteps = timesteps
        self.width = width
        self.height = height
        self.sample_interval = sample_interval
        self.area_scaling = area_scaling
        self.quantity_scaling = quantity_scaling

    def generate_sample(self):
        """
        Single sample generation using the parameters of this wave
        generator.
        :return: The generated sample as numpy array(t, x, y)
        """
        np.set_printoptions(threshold=sys.maxsize)
        
        # --------------- Physical prameters ---------------
        L_x = 1E+6 #* self.area_scaling             # Length of domain in x-direction
        L_y = 1E+6 #* self.area_scaling             # Length of domain in y-direction
        g = 9.81                                    # Acceleration of gravity [m/s^2]
        f_0 = 1E-4                                  # Fixed part of coriolis parameter [1/s]
        beta = 2E-11                                # gradient of coriolis parameter [1/ms]
        rho_0 = 1024.0                              # Density of fluid [kg/m^3)]
        tau_0 = 0.1                                 # Amplitude of wind stress [kg/ms^2]
        use_coriolis = False                        # True if you want coriolis force
        use_friction = False                        # True if you want bottom friction
        use_wind = False                            # True if you want wind stress
        use_beta = False                            # True if you want variation in coriolis
        use_source = False                          # True if you want mass source into the domain
        use_sink = False                            # True if you want mass sink out of the domain

        # --------------- Computational prameters ---------------
        N_x = self.width                     # Number of grid points in x-direction
        N_y = self.height                    # Number of grid points in y-direction
        dx = L_x/(N_x - 1)                   # Grid spacing in x-direction
        dy = L_y/(N_y - 1)                   # Grid spacing in y-direction
        dt = min(dx,dy)/300 #0.1*min(dx, dy)/np.sqrt(g*10) #  # Time step (defined from the CFL condition)
        time_step = 1                        # For counting time loop steps
        max_time_step = self.timesteps * self.sample_interval # Total number of time steps in simulation
        x = np.linspace(-L_x/2, L_x/2, N_x)  # Array with x-points
        y = np.linspace(-L_y/2, L_y/2, N_y)  # Array with y-points
        X, Y = np.meshgrid(x, y)             # Meshgrid for plotting
        X = np.transpose(X)                  # To get plots right
        Y = np.transpose(Y)                  # To get plots right

        # Smooth topography
        H_temp = np.expand_dims(np.linspace(0, 2, N_x), axis=0)
        H_temp = np.repeat(H_temp, H_temp.shape[-1], axis=0)
        H = 100 + np.arctan(H_temp) * 20


        ## Steep topography
        # H = 100 + np.sin(X)*100 # * self.quantity_scaling            # Depth of fluid [m] if two dimensional array, you can decide where shallow where deep
        # print(np.min(H), np.max(H))

        # # Flat H
        # H = 10 + (X * 0)
        # print(np.min(H), np.max(H))



        # Define friction array if friction is enabled.
        if use_friction:
            kappa_0 = 1/(5*24*3600)
            kappa = np.ones((N_x, N_y))*kappa_0

        # Define wind stress arrays if wind is enabled.
        if use_wind:
            tau_x = -tau_0*np.cos(np.pi*y/L_y)*0
            tau_y = np.zeros((1, len(x)))

        # Define coriolis array if coriolis is enabled.
        if use_coriolis:
            if use_beta:
                f = f_0 + beta*y        # Varying coriolis parameter
                L_R = np.sqrt(g*H)/f_0  # Rossby deformation radius
                c_R = beta*g*H/f_0**2   # Long Rossby wave speed
            else:
                f = f_0*np.ones(len(y))                 # Constant coriolis parameter

            alpha = dt*f                # Parameter needed for coriolis scheme
            beta_c = alpha**2/4         # Parameter needed for coriolis scheme


        # Define source array if source is enabled.
        if use_source:
            sigma = np.zeros((N_x, N_y))
            sigma = 0.0001*np.exp(-((X-L_x/2)**2/(2*(1E+5)**2) + (Y-L_y/2)**2/(2*(1E+5)**2)))
            
        # Define source array if source is enabled.
        if use_sink:
            sigma = np.zeros((N_x, N_y))
            w = np.ones((N_x, N_y))*sigma.sum()/(N_x*N_y)

        # ==================================================================================
        # ==================== Allocating arrays and initial conditions ====================
        # ==================================================================================
        u_n = np.zeros((N_x, N_y))      # To hold u at current time step
        u_np1 = np.zeros((N_x, N_y))    # To hold u at next time step
        v_n = np.zeros((N_x, N_y))      # To hold v at current time step
        v_np1 = np.zeros((N_x, N_y))    # To hold v at next time step
        eta_n = np.zeros((N_x, N_y))    # To hold eta at current time step
        eta_np1 = np.zeros((N_x, N_y))  # To hold eta at next time step

        # Temporary variables (each time step) for upwind scheme in eta equation
        # h probably represents eta. Otherwise, it wouldn't make much sense.
        h_e = np.zeros((N_x, N_y)) # h in the east direction
        h_w = np.zeros((N_x, N_y)) # h in the west direction
        h_n = np.zeros((N_x, N_y)) # h in the north direction
        h_s = np.zeros((N_x, N_y)) # h in the south direction
        uhwe = np.zeros((N_x, N_y))# u is the velocity in the x-direction. Hence west and east direction of h is a change in u
        vhns = np.zeros((N_x, N_y))# v is the velocity in the y-direction. Hence north and south direction of h is a change in v

        # Initial conditions for u and v.
        u_n[:, :] = 0.0             # Initial condition for u
        v_n[:, :] = 0.0             # Initial condition for v
        u_n[-1, :] = 0.0            # Ensuring initial u satisfy BC
        v_n[:, -1] = 0.0            # Ensuring initial v satisfy BC

        # Initial condition for eta.
        x_init = 2 * np.random.rand() - 1
        y_init = 2 * np.random.rand() - 1
        
        # Start with a gaussian bump
        gauss_sigma = 0.05E+6
        eta_n = np.exp(-((X-x_init*L_x/2)**2/(2*(gauss_sigma)**2) + (Y-y_init*L_y/2)**2/(2*(gauss_sigma)**2)))
        #eta_n = np.exp(-((X-L_x/2.7)**2/(2*(0.05E+6)**2) + (Y-L_y/4)**2/(2*(0.05E+6)**2)))
        #eta_n[int(3*N_x/8):int(5*N_x/8),int(3*N_y/8):int(5*N_y/8)] = 1.0
        
        # Sampling variable.
        sample = np.zeros((self.timesteps, N_x, N_y))
        # =============== Done with setting up arrays and initial conditions ===============

        t_0 = time.perf_counter()  # For timing the computation loop

        # ==================================================================================
        # ========================= Main time loop for simulation ==========================
        # ==================================================================================
        while time_step <= max_time_step:
            # ------------ Computing values for u and v at next time step --------------
            u_np1[:-1, :] = u_n[:-1, :] + dt * (- g/dx*(eta_n[1:, :] - eta_n[:-1, :])) # subtract each row with the previous row
            v_np1[:, :-1] = v_n[:, :-1] + dt * (- g/dy*(eta_n[:, 1:] - eta_n[:, :-1])) # subtract each column with the previous column
            
            # The original implementation for u_np1 and v_np1
            #u_np1[:-1, :] = u_n[:-1, :] - g*dt/dx*(eta_n[1:, :] - eta_n[:-1, :]) # subtract each row with the previous row
            #v_np1[:, :-1] = v_n[:, :-1] - g*dt/dy*(eta_n[:, 1:] - eta_n[:, :-1]) 
            
            #===================================================================
            # if (time_step == 20):
            #     eta_n += np.exp(-((X-x_init*L_x/2)**2/(2*(0.08E+6)**2) + (Y-y_init*L_y/2)**2/(2*(0.08E+6)**2)))
            #===================================================================
            # Add friction if enabled.
            if use_friction:
                u_np1[:-1, :] -= dt*kappa[:-1, :]*u_n[:-1, :]
                v_np1[:-1, :] -= dt*kappa[:-1, :]*v_n[:-1, :]

            # Add wind stress if enabled.
            if use_wind:
                u_np1[:-1, :] += dt*tau_x[:]/(rho_0*H)
                v_np1[:-1, :] += dt*tau_y[:]/(rho_0*H)

            # Use a corrector method to add coriolis if it's enabled.
            if use_coriolis:
                u_np1[:, :] = (u_np1[:, :] - beta_c*u_n[:, :] + alpha*v_n[:, :])/(1 + beta_c)
                v_np1[:, :] = (v_np1[:, :] - beta_c*v_n[:, :] - alpha*u_n[:, :])/(1 + beta_c)
            
            u_np1[-1, :] = 0.0      # Eastern boundary condition
            v_np1[:, -1] = 0.0      # Northern boundary condition
            
            # -------------------------- Done with u and v -----------------------------

            # --- Computing arrays needed for the upwind scheme in the eta equation.----
            h_e[:-1, :] = np.where(u_np1[:-1, :] > 0, eta_n[:-1, :] + H[:-1, :], eta_n[1:, :] + H[1:, :])
            h_e[-1, :] = eta_n[-1, :] + H[-1, :]

            h_w[0, :] = eta_n[0, :] + H[0, :]
            h_w[1:, :] = np.where(u_np1[:-1, :] > 0, eta_n[:-1, :] + H[:-1, :], eta_n[1:, :] + H[1:, :])

            h_n[:, :-1] = np.where(v_np1[:, :-1] > 0, eta_n[:, :-1] + H[:, :-1], eta_n[:, 1:] + H[:, 1:])
            h_n[:, -1] = eta_n[:, -1] + H[:, -1]

            h_s[:, 0] = eta_n[:, 0] + H[:, 0]
            h_s[:, 1:] = np.where(v_np1[:, :-1] > 0, eta_n[:, :-1] + H[:, :-1], eta_n[:, 1:] + H[:, 1:])

            uhwe[0, :] = u_np1[0, :]*h_e[0, :]
            uhwe[1:, :] = u_np1[1:, :]*h_e[1:, :] - u_np1[:-1, :]*h_w[1:, :]

            vhns[:, 0] = v_np1[:, 0]*h_n[:, 0]
            vhns[:, 1:] = v_np1[:, 1:]*h_n[:, 1:] - v_np1[:, :-1]*h_s[:, 1:]
            
            # ------------------------- Upwind computations done -------------------------
            

            # ----------------- Computing eta values at next time step -------------------
            eta_np1[:, :] = eta_n[:, :] - dt*(uhwe[:, :]/dx + vhns[:, :]/dy)    # Without source/sink

            # Add source term if enabled.
            if use_source:
                eta_np1[:, :] += dt*sigma

            # Add sink term if enabled.
            if use_sink:
                eta_np1[:, :] -= dt*w
            # ----------------------------- Done with eta --------------------------------
            
            # Store eta and (u, v) every anim_interval time step for animations.
            if time_step % self.sample_interval == 0:
                if __name__ == '__main__':
                    print("Total Quantity: {}".format(np.sum(eta_n)))

                sample[int(time_step/self.sample_interval) - 1,:,:] = eta_n
            
            u_n = np.copy(u_np1)        # Update u for next iteration
            v_n = np.copy(v_np1)        # Update v for next iteration
            eta_n = np.copy(eta_np1)    # Update eta for next iteration
            
            time_step += 1

        return sample

if __name__ == '__main__':
    
    sim = Simulator(150, 150, 150)
    sim.generate_sample()
