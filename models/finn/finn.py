"""
Finite Volume Neural Network implementation with PyTorch.
"""

import torch.nn as nn
import torch as th
from torchdiffeq import odeint
# from pylint.checkers.variables import in_for_else_branch
import utils.solvers as solver
import matplotlib.pyplot as plt


class FINN(nn.Module):
    """
    This is the parent FINN class. This class initializes all sharable parameters
    between different implementations to be inherited to each of the implementation.
    It also contains the initialization of the function_learner and reaction_learner
    NN which learns the constitutive relationships (or the flux multiplier) and
    reaction functions, respectively.
    """
    
    def __init__(self, u, D, BC, layer_sizes, device, mode,
                 config, learn_coeff, learn_BC, train_mini_batch, learn_stencil, bias, sigmoid):
        """
        Constructor.
        
        Inputs:
        :param u: the unknown variable
        :type u: th.tensor[len(t), Nx, Ny, num_vars]
        
        :param D: diffusion coefficient
        :type D: np.array[num_vars] --- th.tensor is also accepted
        
        :param BC: the boundary condition values. In case of Dirichlet BC, this
        contains the scalar values. In case of Neumann, this contains the flux
        values.
        :type BC: np.array[num_bound, num_vars] --- th.tensor is also accepted
        
        :param layer_sizes: a list of the hidden nodes for each layer (including
        the input and output features)
        :type layer_sizes: list[num_hidden_layers + 2]
        
        :param device: the device to perform simulation
        :type device: th.device
        
        :param mode: mode of simulation ("train" or "test")
        :type mode: str
        
        :param config: configuration of simulation parameters
        :type config: dict
        
        :param learn_coeff: a switch to set diffusion coefficient to be learnable
        :type learn_coeff: bool
        
        :param learn_BC: a switch to set boundary condition to be learnable
        :type learn_BC: bool
        
        :param train_mini_batch: a switch to set mini-batch training
        :type learn_mini_batch: bool
        
        :param learn_stencil: a switch to set the numerical stencil to be learnable
        :type learn_stencil: bool
        
        :param bias: a bool value to determine whether to use bias values in the function_learner
        :type bias bool
        
        :param sigmoid: a bool value to determine whether to use sigmoid at the
        output layer
        :type sigmoid: bool
        
        Output:
        :return: the full field solution of u from time t0 until t_end
        :rtype: th.tensor[len(t), Nx, Ny, num_vars]

        """
        
        super(FINN, self).__init__()
        
        self.device = device
        self.Nx = u.size()[1]
        self.layer_sizes = layer_sizes
        self.mode = mode
        self.cfg = config
        self.bias = bias
        self.sigmoid = sigmoid
        self.train_mini_batch = train_mini_batch
        self.learn_BC = learn_BC
        
        if not learn_BC:
            self.BC = th.tensor(BC, dtype=th.float).to(device=self.device)
        else:
            self.BC = nn.Parameter(th.tensor(BC, dtype=th.float))
            
        if not learn_coeff:
            self.D = th.tensor(D, dtype=th.float).to(device=self.device)
        else:
            self.D = nn.Parameter(th.tensor(D, dtype=th.float))
        
        if not learn_stencil:
            self.stencil = th.tensor([-1.0, 1.0], dtype=th.float).to( # it was [-500000.0, 500000.0]
                device=self.device)
        else:
            self.stencil = th.tensor(
                [th.normal(th.tensor([-1.0]), th.tensor([0.1])),
                 th.normal(th.tensor([1.0]), th.tensor([0.1]))],
                dtype=th.float)
            self.stencil = nn.Parameter(self.stencil)
            

    def function_learner(self):
        """
        This function constructs a feedforward NN required for calculation
        of constitutive function (or flux multiplier) as a function of u.
        """
        layers = list()
         
        for layer_idx in range(len(self.layer_sizes) - 1):
            # Add a fully connected layer
            layer = nn.Linear(
                in_features=self.layer_sizes[layer_idx],
                out_features=self.layer_sizes[layer_idx + 1],
                bias=self.bias,
                dtype=th.float64
                ).to(device=self.device)
            layers.append(layer)


            # Apply activation function
            if layer_idx < len(self.layer_sizes) - 2:
                layers.append(nn.Tanh())
            elif self.sigmoid:
                # Use sigmoid function to keep the values strictly positive
                # (all outputs have the same sign)
                layers.append(nn.Sigmoid())

        print(nn.Sequential(*nn.ModuleList(layers)))
        return nn.Sequential(*nn.ModuleList(layers))


    def function_learner2(self):
        """
        This function constructs a feedforward NN required for calculation
        of constitutive function (or flux multiplier) as a function of u.
        """
        layers = list()

        for layer_idx in range(len(self.layer_sizes) - 1):
            # Add a fully connected layer
            if layer_idx == len(self.layer_sizes) - 2:
                layer = nn.Linear(
                    in_features=self.layer_sizes[layer_idx],
                    out_features=self.layer_sizes[layer_idx + 1],
                    bias=False,
                    dtype=th.float64
                ).to(device=self.device)
                layers.append(layer)
            else:
                layer = nn.Linear(
                    in_features=self.layer_sizes[layer_idx],
                    out_features=self.layer_sizes[layer_idx + 1],
                    bias=self.bias,
                    dtype=th.float64
                ).to(device=self.device)
                layers.append(layer)

            # Apply activation function
            if layer_idx < len(self.layer_sizes) - 2:
                layers.append(nn.Tanh())
            elif self.sigmoid:
                # Use sigmoid function to keep the values strictly positive
                # (all outputs have the same sign)
                layers.append(nn.Sigmoid())

        print(nn.Sequential(*nn.ModuleList(layers)))
        return nn.Sequential(*nn.ModuleList(layers))

    def function_learner_cnn(self):
        """
        This function constructs a convolutional NN required for calculation
        of u[t+1] as a function of u[t]
        """

        conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=100,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False, dtype=th.double).to(device=self.device)
        conv2 = nn.Conv2d(
            in_channels=100,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False, dtype=th.double).to(device=self.device)

        conv3 = nn.Conv2d(
            in_channels=100,
            out_channels=100,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False, dtype=th.double).to(device=self.device)

        conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False, dtype=th.double).to(device=self.device)

        layers = list((conv1, nn.ReLU(), conv2, nn.Tanh())) # conv3, nn.ReLU(), conv4, nn.Tanh()))
        print(nn.Sequential(*nn.ModuleList(layers)))

        return nn.Sequential(*nn.ModuleList(layers))


class FINN_Burger(FINN):
    """
    This is the inherited FINN class for the Burgers equation implementation.
    This class inherits all parameters from the parent FINN class.
    """
    
    def __init__(self, u, D, BC, dx, layer_sizes, device, mode="train",
                 config=None, learn_coeff=False, learn_BC=False, train_mini_batch=False,
                 learn_stencil=False, bias=False, sigmoid=False):
        """
        Constructor.
        
        Inputs:
        Same with the parent FINN class, with the addition of dx (the spatial resolution)
        """
        
        super().__init__(u, D, BC, layer_sizes, device, mode, config, learn_coeff,
                         learn_BC, train_mini_batch, learn_stencil, bias, sigmoid)
        
        self.dx = dx
        
        # Initialize the function_learner to learn the first order flux multiplier
        self.func_nn = self.function_learner().to(device=self.device)
    
    """
    TODO: Implement flux kernel for test (if different BC is used)
    """
        
    def flux_kernel(self, t, u):
        """
        This function defines the flux kernel for training, which takes ui and its
        neighbors as inputs, and returns the integrated flux approximation (up to
        second order derivatives)
        """
        
        # Approximate the first order flux multiplier
        a = self.func_nn(u.unsqueeze(-1))

        # Apply the ReLU function for upwind scheme to prevent numerical
        # instability
        a_plus = th.relu(a[...,0])
        a_min = -th.relu(-a[...,0])


        ## Calculate fluxes at the left boundary of control volumes i
        
        # Calculate the flux at the left domain boundary
        if not self.train_mini_batch:
            left_bound_flux = (self.D*(self.stencil[0]*u[0] +
                                self.stencil[1]*self.BC[0,0]) -
                                a_plus[0]/self.dx*(-self.stencil[0]*u[0] -
                                self.stencil[1]*self.BC[0,0]))
        else:
            left_bound_flux = (self.D*(self.stencil[0]*u[0] +
                                self.stencil[1]*self.BC[:,0,0]) -
                                a_plus[0]/self.dx*(-self.stencil[0]*u[0] -
                                self.stencil[1]*self.BC[:,0,0]))               
        # Calculate the fluxes between control volumes i and their left neighbors
        left_neighbors = self.D*(self.stencil[0]*u[1:] +
                            self.stencil[1]*u[:-1]) -\
                            a_plus[1:]/self.dx*(-self.stencil[0]*u[1:] -
                            self.stencil[1]*u[:-1])
        
        # Concatenate the left fluxes
        if not self.train_mini_batch:
            left_flux = th.cat((left_bound_flux, left_neighbors))
        else:
            left_flux = th.cat((left_bound_flux.unsqueeze(0), left_neighbors))
        
        
        ## Calculate fluxes at the right boundary of control volumes i
        
        # Calculate the flux at the right domain boundary
        if not self.train_mini_batch:
            right_bound_flux = (self.D*(self.stencil[0]*u[-1] +
                                self.stencil[1]*self.BC[1,0]) -
                                a_min[-1]/self.dx*(self.stencil[0]*u[-1] +
                                self.stencil[1]*self.BC[1,0]))
        else:
            right_bound_flux = (self.D*(self.stencil[0]*u[-1] +
                                self.stencil[1]*self.BC[:,1,0]) -
                                a_min[-1]/self.dx*(self.stencil[0]*u[-1] +
                                self.stencil[1]*self.BC[:,1,0]))
        
                 
        # Calculate the fluxes between control volumes i and their right neighbors
        right_neighbors = self.D*(self.stencil[0]*u[:-1] +
                            self.stencil[1]*u[1:]) -\
                            a_min[:-1]/self.dx*(self.stencil[0]*u[:-1] +
                            self.stencil[1]*u[1:])
                        
        
        # Concatenate the right fluxes
        if not self.train_mini_batch:
            right_flux = th.cat((right_neighbors, right_bound_flux))
        else:
            right_flux = th.cat((right_neighbors, right_bound_flux.unsqueeze(0)))
        
        
        # Integrate the fluxes at all boundaries of control volumes i
        flux = left_flux + right_flux

        return flux
    
    def state_kernel(self, t, u):
        """
        This function defines the state kernel for training, which takes the
        fluxes as inputs, and returns du/dt)
        """
        
        flux = self.flux_kernel(t, u)
        
        # Since there is no reaction term to be learned, du/dt = fluxes
        state = flux
        
        return state
    
    
    def forward(self, t, u):
        """
        This function integrates du/dt through time using the Neural ODE method
        """
        
        # The odeint function receives the function state_kernel that calculates
        # du/dt, the initial condition u[0], and the time at which the values of
        # u will be saved t
        method = "dopri5" # rk4, dopri5
        ## Change u[0] to u for online learning.
        if method == "dopri5":
            pred = odeint(self.state_kernel, u[0], t, method="dopri5")
        elif method == "rk4":
            pred = solver.integrate(self.state_kernel, u[0], t, "rk4")
        elif method == "euler":
            pred = solver.integrate(self.state_kernel, u[0], t, "euler")
            
        return pred
    

class FINN_AllenCahn(FINN):
    """
    This is the inherited FINN class for the Burger equation implementation.
    This class inherits all parameter from the parent FINN class.
    """
    
    def __init__(self, u, D, BC, dx, layer_sizes, device, mode="train",
                 config=None, learn_coeff=False, learn_BC=False, train_mini_batch=False, learn_stencil=False,
                 bias=False, sigmoid=False):
        """
        Constructor.
        
        Inputs:
        Same with the parent FINN class, with the addition of dx (the spatial resolution)
        """
        
        super().__init__(u, D, BC, layer_sizes, device, mode, config, learn_coeff, learn_BC, train_mini_batch,
                         learn_stencil, bias, sigmoid)
        
        self.dx = dx
        
        # Initialize the function_learner to learn the first order flux multiplier
        self.func_nn = self.function_learner().to(device=self.device)
        
        # Initialize the multiplier of the retardation factor function (denormalization)
        # self.p_mult = nn.Parameter(th.tensor([10.0],dtype=th.float))
    
    """
    TODO: Implement flux kernel for test (if different BC is used)
    """
        
    def flux_kernel(self, t, u):
        """
        This function defines the flux kernel for training, which takes ui and its
        neighbors as inputs, and returns the integrated flux approximation (up to
        second order derivatives)
        """
        
        ## Calculate fluxes at the left boundary of control volumes i

        # Calculate the flux at the left domain boundary
        if not self.train_mini_batch:
            left_bound_flux = self.D*10*(self.stencil[0]*u[0] +
                                self.stencil[1]*self.BC[0, 0])
        else:
            left_bound_flux = self.D*10*(self.stencil[0]*u[0] +
                                self.stencil[1]*self.BC[:,0, 0])
                            
        # Calculate the fluxes between control volumes i and their left neighbors
        left_neighbors = self.D*10*(self.stencil[0]*u[1:] +
                            self.stencil[1]*u[:-1])
        
        # Concatenate the left fluxes
        if not self.train_mini_batch:
            left_flux = th.cat((left_bound_flux, left_neighbors))
        else:
            left_flux = th.cat((left_bound_flux.unsqueeze(0), left_neighbors))
        
        ## Calculate fluxes at the right boundary of control volumes i
        
        # Calculate the flux at the right domain boundary
        if not self.train_mini_batch:
            right_bound_flux = self.D*10*(self.stencil[0]*u[-1] +
                                self.stencil[1]*self.BC[1, 0])
        else:
            right_bound_flux = self.D*10*(self.stencil[0]*u[-1] +
                                self.stencil[1]*self.BC[:,1, 0])
                 
        # Calculate the fluxes between control volumes i and their right neighbors
        right_neighbors = self.D*10*(self.stencil[0]*u[:-1] +
                            self.stencil[1]*u[1:])
        
        # Concatenate the right fluxes
        if not self.train_mini_batch:
            right_flux = th.cat((right_neighbors, right_bound_flux))
        else:
            right_flux = th.cat((right_neighbors, right_bound_flux.unsqueeze(0)))
            
        
        # Integrate the fluxes at all boundaries of control volumes i
        flux = left_flux + right_flux
        
        return flux
    
    def state_kernel(self, t, u):
        """
        This function defines the state kernel for training, which takes the
        fluxes as inputs, and returns du/dt)
        """
        
        flux = self.flux_kernel(t, u)
        
        # Since there is no reaction term to be learned, du/dt = fluxes
        state = flux + self.func_nn(u.unsqueeze(-1)).squeeze() #*self.p_mult
        
        return state
    
    def forward(self, t, u):
        """
        This function integrates du/dt through time using the Neural ODE method
        """
        
        # The odeint function receives the function state_kernel that calculates
        # du/dt, the initial condition u[0], and the time at which the values of
        # u will be saved t
        pred = odeint(self.state_kernel, u[0], t)
        
        return pred
    
    

class FINN_DiffReact(FINN):
    """
    This is the inherited FINN class for the diffusion-reaction equation implementation.
    This class inherits all parameter from the parent FINN class.
    """
    def __init__(self, u, D, BC, dx, dy, layer_sizes, device, mode="train",
                 config=None, learn_coeff=False, learn_BC=False, train_mini_batch=False,
                 learn_stencil=False, bias=False, sigmoid=False):
        
        
        super().__init__(u, D, BC, layer_sizes, device, mode, config, learn_coeff, learn_BC, train_mini_batch,
                         learn_stencil, bias, sigmoid)
        
        """
        Constructor.
        
        Inputs:
        Same with the parent FINN class, with the addition of dx and dy (the
        spatial resolution)
        """
        
        self.dx = dx
        self.dy = dy
        
        self.Ny = u.size()[2]
        
        # Initialize the reaction_learner to learn the reaction term
        self.func_nn = self.function_learner().to(device=self.device)

        self.right_flux = th.zeros(49, 49, 2, device=self.device)
        self.left_flux = th.zeros(49, 49, 2, device=self.device)
 
        self.bottom_flux = th.zeros(49, 49, 2, device=self.device)
        self.top_flux = th.zeros(49, 49, 2, device=self.device)
        
    
    """
    TODO: Implement flux kernel for test (if different BC is used)
    """
        
    def flux_kernel(self, t, u):
        """
        This function defines the flux kernel for training, which takes ui and its
        neighbors as inputs, and returns the integrated flux approximation (up to
        second order derivatives)
        """
        
        # Separate u into act and inh
        act = u[...,0]
        inh = u[...,1]
        
        ## Calculate fluxes at the left boundary of control volumes i
        
        ## For act
        # Calculate the flux at the left domain boundary
        left_bound_flux_act = th.cat(self.Ny * [self.BC[0,0].unsqueeze(0)]).unsqueeze(0)
                            
        # Calculate the fluxes between control volumes i and their left neighbors
        left_neighbors_act = self.D[0]*(self.stencil[0]*act[1:] +
                            self.stencil[1]*act[:-1])

        # Concatenate the left fluxes
        left_flux_act = th.cat((left_bound_flux_act, left_neighbors_act))
                
        ## For inh
        # Calculate the flux at the left domain boundary
        left_bound_flux_inh = th.cat(self.Ny * [self.BC[0,1].unsqueeze(0)]).unsqueeze(0)
                
        # Calculate the fluxes between control volumes i and their left neighbors
        left_neighbors_inh = self.D[1]*(self.stencil[0]*inh[1:] +
                            self.stencil[1]*inh[:-1])

        # Concatenate the left fluxes
        left_flux_inh = th.cat((left_bound_flux_inh, left_neighbors_inh))

        # Stack the left fluxes of act and inh together
        left_flux = th.stack((left_flux_act, left_flux_inh), dim=len(act.size()))

        ## Calculate fluxes at the right boundary of control volumes i
        
        ## For act
        # Calculate the flux at the right domain boundary
        right_bound_flux_act = th.cat(self.Ny * [self.BC[1,0].unsqueeze(0)]).unsqueeze(0)
                            
        # Calculate the fluxes between control volumes i and their right neighbors 
        right_neighbors_act = self.D[0]*(self.stencil[0]*act[:-1] +
                            self.stencil[1]*act[1:])
        
        # Concatenate the right fluxes
        right_flux_act = th.cat((right_neighbors_act, right_bound_flux_act))
        
        ## For inh
        # Calculate the flux at the right domain boundary  
        right_bound_flux_inh = th.cat(self.Ny * [self.BC[1,1].unsqueeze(0)]).unsqueeze(0)
           
        # Calculate the fluxes between control volumes i and their right neighbors  
        right_neighbors_inh = self.D[1]*(self.stencil[0]*inh[:-1] +
                            self.stencil[1]*inh[1:])

        # Concatenate the right fluxes
        right_flux_inh = th.cat((right_neighbors_inh, right_bound_flux_inh))
        
        # Stack the right fluxes of act and inh together
        right_flux = th.stack((right_flux_act, right_flux_inh), dim=len(act.size()))
        
        
        ## Calculate fluxes at the bottom boundary of control volumes i
        
        ## For act
        # Calculate the flux at the bottom domain boundary
        bottom_bound_flux_act = th.cat(self.Nx * [self.BC[2,0].unsqueeze(0)]).unsqueeze(-1)
           
        # Calculate the fluxes between control volumes i and their bottom neighbors                   
        bottom_neighbors_act = self.D[0]*(self.stencil[0]*act[:,1:] +
                            self.stencil[1]*act[:,:-1])
        
        # Concatenate the bottom fluxes
        bottom_flux_act = th.cat((bottom_bound_flux_act, bottom_neighbors_act), dim=1)
        
        ## For inh
        # Calculate the flux at the bottom domain boundary
        bottom_bound_flux_inh = th.cat(self.Nx * [self.BC[2,1].unsqueeze(0)]).unsqueeze(-1)
                            
        # Calculate the fluxes between control volumes i and their bottom neighbors
        bottom_neighbors_inh = self.D[1]*(self.stencil[0]*inh[:,1:] +
                            self.stencil[1]*inh[:,:-1])
        
        # Concatenate the bottom fluxes
        bottom_flux_inh = th.cat((bottom_bound_flux_inh, bottom_neighbors_inh),dim=1)

        # Stack the bottom fluxes of act and inh together
        bottom_flux = th.stack((bottom_flux_act, bottom_flux_inh), dim=len(act.size()))

        ## Calculate fluxes at the top boundary of control volumes i
        
        ## For act
        # Calculate the flux at the top domain boundary
        top_bound_flux_act = th.cat(self.Nx * [self.BC[3,0].unsqueeze(0)]).unsqueeze(-1)
                            
        # Calculate the fluxes between control volumes i and their top neighbors
        top_neighbors_act = self.D[0]*(self.stencil[0]*act[:,:-1] +
                            self.stencil[1]*act[:,1:])
        
        # Concatenate the top fluxes
        top_flux_act = th.cat((top_neighbors_act, top_bound_flux_act), dim=1)
        
        ## For inh
        # Calculate the flux at the top domain boundary
        top_bound_flux_inh = th.cat(self.Nx * [self.BC[3,1].unsqueeze(0)]).unsqueeze(-1)
                  
        # Calculate the fluxes between control volumes i and their top neighbors
        top_neighbors_inh = self.D[1]*(self.stencil[0]*inh[:,:-1] +
                            self.stencil[1]*inh[:,1:])
        
        # Concatenate the top fluxes
        top_flux_inh = th.cat((top_neighbors_inh, top_bound_flux_inh), dim=1)
        
        # Stack the top fluxes of act and inh together
        top_flux = th.stack((top_flux_act, top_flux_inh), dim=len(act.size()))
        
        # Integrate the fluxes at all boundaries of control volumes i
        flux = left_flux + right_flux + bottom_flux + top_flux
        
        return flux
    
    def state_kernel(self, t, u):
        """
        This function defines the state kernel for training, which takes the
        fluxes as inputs, and returns du/dt)
        """
        
        flux = self.flux_kernel(t, u)
        
        # Add the reaction term to the fluxes term to obtain du/dt
        state = flux + self.func_nn(u)
        
        return state
    
    def forward(self, t, u):
        """
        This function integrates du/dt through time using the Neural ODE method
        """
        
        # The odeint function receives the function state_kernel that calculates
        # du/dt, the initial condition u[0], and the time at which the values of
        # u will be saved t
                
        pred = odeint(self.state_kernel, u[0], t)
        
        return pred
    
    

class FINN_ShallowWater(FINN):
    """
    This is the inherited FINN class for the Shallow-Water equation implementation.
    This class inherits all parameter from the parent FINN class.
    """
    def __init__(self, u, D, BC, dx, dy, layer_sizes, device, mode="train",
                 config=None, learn_coeff=False, learn_BC=False, train_mini_batch=False,
                 learn_stencil=False, bias=False, sigmoid=False):

        """
        Constructor.

        Inputs:
        Same with the parent FINN class, with the addition of dx and dy (the
        spatial resolution)
        """
        
        super().__init__(u, D, BC, layer_sizes, device, mode, config, learn_coeff, learn_BC,
                         train_mini_batch, learn_stencil, bias, sigmoid)


        self.dx = dx
        self.dy = dy

        self.Nx = u.size()[1]
        self.Ny = u.size()[2]
        
        # Initialize the reaction_learner to learn the reaction term
        self.func_nn = self.function_learner().to(device=self.device)
        self.func_nn2 = self.function_learner2().to(device=self.device)
        # self.func_cnn = self.function_learner_cnn().to(device=self.device)
        
        self.g = 9.81 #nn.Parameter(th.tensor(9.81, dtype=th.float64))
        # self.H = nn.Parameter(th.zeros((self.Nx, self.Ny), dtype=th.float64)+110)

        ## Smooth topography
        H_temp = th.linspace(0, 2, self.Nx).unsqueeze(0)
        H_temp = th.repeat_interleave(H_temp, H_temp.shape[-1], dim=0)
        self.H = 100 + th.atan(H_temp) * 20

        
    def flux_kernel(self, t, u, eq):
        """
        This function defines the flux kernel for training, which takes ui and its
        neighbors as inputs, and returns the integrated flux approximation (up to
        second order derivatives)
        """

        if eq == 'velo':

            #### Solve for velocity functions ####

            # Extract eta[t] from the data
            eta = u[..., 0].type(th.float64)

            # ## Block to compute velos
            # # Eastern Boundary Condition
            # bound_velo_x = (self.BC[0, 0] * th.ones_like(eta[-1, :])).unsqueeze(0)
            #
            # # Calculate the flux between control volumes i and their right neighbors
            # neighbors_velo_x = - self.g / self.dx * (eta[1:, :] - eta[:-1, :])
            #
            # # Concatenate the velo_x fluxes
            # dvelo_x_dt = th.cat((neighbors_velo_x, bound_velo_x))
            #
            # # Northern Boundary Condition
            # bound_velo_y = (self.BC[1, 0] * th.ones_like(eta[:, -1])).unsqueeze(-1)
            #
            # # Calculate the flux between control volumes i and their up neighbors
            # neighbors_velo_y = - self.g / self.dy * (eta[:, 1:] - eta[:, :-1])
            #
            # # Concatenate the velo_y fluxes
            # dveloy_dt = th.cat((neighbors_velo_y, bound_velo_y), dim=1)
            #
            # # Stack eta and the time derivatives of the velocity equations
            # flux = th.stack((eta, dvelo_x_dt, dveloy_dt), dim=-1)

            # ## Block to approximate velos
            # Eastern Boundary Condition
            bound_velo_x = (self.BC[0, 0] * th.ones_like(eta[-1, :])).unsqueeze(0)

            # Apply padding
            # eta1 = th.cat((eta, bound_velo_x))  # [33, 32]

            # Approximate velo_x
            eta_x = th.stack((eta[1:, :],  eta[:-1, :]), dim=-1).type(th.float64)

            neighbors_velo_x = self.func_nn(eta_x).squeeze()

            # Concatenate velo_x fluxes
            dvelo_x_dt = th.cat((neighbors_velo_x, bound_velo_x))


            # Southern Boundary Condition
            bound_velo_y = (self.BC[1, 0] * th.ones_like(eta[:, -1])).unsqueeze(-1)

            # Apply padding
            # eta2 = th.cat((eta, bound_velo_y), dim=1)  # [32, 33]

            # Approximate velo_y
            eta_y = th.stack((eta[:, 1:], eta[:, :-1]), dim=-1).type(th.float64)
            neighbors_velo_y = self.func_nn(eta_y).squeeze()

            # Concatenate velo_y fluxes
            dvelo_y_dt = th.cat((neighbors_velo_y, bound_velo_y), dim=1)

            # Stack eta and the time derivatives of the velocity equations
            flux = th.stack((eta, dvelo_x_dt, dvelo_y_dt), dim=-1)

            return flux


        elif eq == 'eta':

            #### Solve for eta equation ####

            # --- Computing arrays needed for the upwind scheme in the eta equation.---
            # Temporary variables (each time step) for upwind scheme in eta equation
            h_e = th.zeros((self.Nx, self.Ny))
            h_w = th.zeros((self.Nx, self.Ny))
            h_n = th.zeros((self.Nx, self.Ny))
            h_s = th.zeros((self.Nx, self.Ny))
            uhwe = th.zeros((self.Nx, self.Ny))
            vhns = th.zeros((self.Nx, self.Ny))

            # Extract the corresponding function values from the input
            eta    = u[..., 0]
            velo_x = u[..., 1]
            velo_y = u[..., 2]

            # --- Computing arrays needed for the upwind scheme in the eta equation.----
            # h_e[:-1, :] = th.where(velo_x[:-1, :] > 0, eta[:-1, :] + self.H, eta[1:, :] + self.H)
            # h_e[-1, :] = eta[-1, :] + self.H

            h_stack = th.stack((eta[:-1, :], eta[1:, :]), dim=-1).type(th.float64)
            h = self.func_nn2(h_stack).squeeze()
            h_e[:-1, :] = h + self.H[:-1, :]
            h_e[-1, :] = eta[-1, :] + self.H[-1, :]

            # h_w[0, :] = eta[0, :] + self.H
            # h_w[1:, :] = th.where(velo_x[:-1, :] > 0, eta[:-1, :] + self.H, eta[1:, :] + self.H)

            h_w[1:, :] = h + self.H[1:, :]
            h_w[0, :] = eta[0, :] + self.H[0, :]

            # h_n[:, :-1] = th.where(velo_y[:, :-1] > 0, eta[:, :-1] + self.H, eta[:, 1:] + self.H)
            # h_n[:, -1] = eta[:, -1] + self.H

            v_stack = th.stack((eta[:, :-1], eta[:, 1:]), dim=-1).type(th.float64)
            v = self.func_nn2(v_stack).squeeze()
            h_n[:, :-1] = v + self.H[:, :-1]
            h_n[:, -1] = eta[:, -1] + self.H[:, -1]

            # h_s[:, 0] = eta[:, 0] + self.H
            # h_s[:, 1:] = th.where(velo_y[:, :-1] > 0, eta[:, :-1] + self.H, eta[:, 1:] + self.H)

            h_s[:, 0] = eta[:, 0] + self.H[:, 0]
            h_s[:, 1:] = v + self.H[:, 1:]

            uhwe[0, :]  = velo_x[0, :] * h_e[0, :]
            uhwe[1:, :] = velo_x[1:, :] * h_e[1:, :] - velo_x[:-1, :] * h_w[1:, :]

            vhns[:, 0]  = velo_y[:, 0] * h_n[:, 0]
            vhns[:, 1:] = velo_y[:, 1:] * h_n[:, 1:] - velo_y[:, :-1] * h_s[:, 1:]
            # ------------------------- Upwind computations done -------------------------

            deta_dt = - (uhwe / self.dx + vhns / self.dy)  # Without source/sink

            return deta_dt

    def state_kernel(self, t, u, eq):
        """
        This function defines the state kernel for training, which takes the
        fluxes as inputs, and returns du/dt)
        """
        
        flux = self.flux_kernel(t, u, eq)
        
        # Since there is no reaction term to be learned, du/dt = fluxes
        state = flux
        
        return state
    
    def forward(self, t, u):
        """
        This function integrates du/dt through time using the Neural ODE method
        """
        
        # The odeint function receives the function state_kernel that calculates
        # du/dt, the initial condition u[0], and the time at which the values of
        # u will be saved t
        
        method = "euler" # rk4, dopri5
        if method == "dopri5":
            pred = odeint(self.state_kernel, u[0], t, method="euler")
        elif method == "rk4":
            pred = solver.integrate(self.state_kernel, u[0], t, "rk4")
        elif method == "euler":
            pred = solver.integrate(self.state_kernel, u[0], t, "euler")

        return pred