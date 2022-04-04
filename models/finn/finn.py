"""
Finite Volume Neural Network implementation with PyTorch.
"""

import torch.nn as nn
import torch as th
from torchdiffeq import odeint
from pylint.checkers.variables import in_for_else_branch


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
            self.stencil = th.tensor([-1.0, 1.0], dtype=th.float).to(
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
            layer = nn.Linear(
                in_features=self.layer_sizes[layer_idx],
                out_features=self.layer_sizes[layer_idx + 1],
                bias=self.bias
                ).to(device=self.device)
            layers.append(layer)
         
            if layer_idx < len(self.layer_sizes) - 2:
                layers.append(nn.Tanh())
            elif self.sigmoid:
                # Use sigmoid function to keep the values strictly positive
                # (all outputs have the same sign)
                layers.append(nn.Sigmoid())
         
        return nn.Sequential(*nn.ModuleList(layers))
    
        
class FINN_Burger(FINN):
    """
    This is the inherited FINN class for the Allen-Cahn equation implementation.
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
                                self.stencil[1]*self.BC[0,0]) -\
                                a_plus[0]/self.dx*(-self.stencil[0]*u[0] -
                                self.stencil[1]*self.BC[0,0]))
        else:
            left_bound_flux = (self.D*(self.stencil[0]*u[0] +
                                self.stencil[1]*self.BC[:,0,0]) -\
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
                                self.stencil[1]*self.BC[1,0]) -\
                                a_min[-1]/self.dx*(self.stencil[0]*u[-1] +
                                self.stencil[1]*self.BC[1,0]))
        else:
            right_bound_flux = (self.D*(self.stencil[0]*u[-1] +
                                self.stencil[1]*self.BC[:,1,0]) -\
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
        pred = odeint(self.state_kernel, u[0], t)
        
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