#! env/bin/python3

"""
Main file for testing (evaluating) a FINN model
"""

import numpy as np
import torch as th
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import sys

sys.path.append("..")
from utils.configuration import Configuration
from finn import FINN_Burger, FINN_AllenCahn


def run_testing(visualize=True, model_number=None, infer_BC=False):

    # Load the user configurations
    config = Configuration("config.json")
    
    # Append the model number to the name of the model
    if model_number is None:
        model_number = config.model.number
    config.model.name = config.model.name + "_" + str(model_number).zfill(2)

    # Print some information to console
    print("Model name:", config.model.name)

    # Hide the GPU(s) in case the user specified to use the CPU in the config
    # file
    if config.general.device == "CPU":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    root_path = os.path.abspath("../../data")
    data_path = os.path.join(root_path, config.data.type, config.data.name)
    
    # Set device on GPU if specified in the configuration file, else CPU
    device = th.device(config.general.device)
    
    if config.data.type == "burger":
        # Load samples, together with x, y, and t series
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
        x = np.load(os.path.join(data_path, "x_series.npy"))
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                             dtype=th.float).to(device=device)
        print(u.shape)
        
        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)
        
        dx = x[1]-x[0]
    
        # Initialize and set up the model
        model = FINN_Burger(
            u = u,
            D = np.array([0.01/np.pi/dx**2]),
            BC = np.array([[0.0], [0.0]]), 
            dx = dx,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="test",
            learn_coeff=False,
            learn_BC=True,
            train_mini_batch=False
        ).to(device=device)
        
    elif config.data.type == "allen_cahn":
        # Load samples, together with x, y, and t series
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
        x = np.load(os.path.join(data_path, "x_series.npy"))
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                             dtype=th.float).to(device=device)
        
        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)
        
        dx = x[1]-x[0]
    
        # Initialize and set up the model
        model = FINN_AllenCahn(
            u = u,
            D = np.array([0.005/dx**2]),
            BC = np.array([[0.0], [0.0]]),
            dx = dx,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="test",
            learn_coeff=False,
            learn_BC=True,
            train_mini_batch=False
        ).to(device=device)
    

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Trainable model parameters: {pytorch_total_params}\n")

    #Load the trained weights from the checkpoints into the model
    model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                               "checkpoints",
                                               config.model.name,
                                               config.model.name+".pt")))
     
    
    
    # Initialize the criterion (loss)
    criterion = nn.MSELoss()
    
    
    '''  INFERENCE  '''
    
    # Block for inferring the BC-Values
    if infer_BC:
        # Set all trainable parameters to false
        for p in model.parameters():
            p.requires_grad = False
        
        if config.data.type == "burger":
            # Set BC as parameter
            model.BC = nn.Parameter(th.tensor([[np.random.uniform(-1,1)], [np.random.uniform(-1,1)]], dtype=th.float))
            #[[np.random.uniform(-1,1)], [np.random.uniform(-1,1)]]
            print(f"BC: {model.BC}")
        
        elif config.data.type == "allen_cahn":
            # Set BC as parameter
            model.BC = nn.Parameter(th.tensor([[np.random.uniform(-0.3,0.3)], [np.random.uniform(-0.3,0.3)]], dtype=th.float))
            print(f"BC: {model.BC}")
        
        # Count number of trainable parameters
        pytorch_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
            )
        print(f"Trainable model parameters: {pytorch_total_params}\n")
        
        # Set up an optimizer
        optimizer = th.optim.Adam(model.parameters(), lr=config.training.learning_rate)
        
        #
        # Set up lists to save and store the epoch errors
        epoch_errors_train = []
        best_train = np.infty
        
        # The amount of the test data for inferring
        data_infer = 30
        
        # Set the number of iteration
        iterations = 700
        
        # Slice data for the inference
        u_infer = u[:data_infer,:]
        
        # Initialize lists to store
        left_BC = []
        right_BC = []
        
        left_BC_grad = []
        right_BC_grad = []
        
        mse_inference = []
        
        for i in range(iterations):
            # Define the closure function that consists of resetting the
            # gradient buffer, loss function calculation, and backpropagation
            # It is necessary for LBFGS optimizer, because it requires multiple
            # function evaluations
            def closure():
                # Set the model to train mode
                model.train()
                    
                # Reset the optimizer to clear data from previous iterations
                optimizer.zero_grad()
    
                # Forward propagate and calculate loss function
                u_hat_infer = model(t=t[:data_infer], u=u_infer)
    
                mse = criterion(u_hat_infer, u_infer)
                
                mse.backward()
                
                return mse
        
            optimizer.step(closure)
                
            # Extract the MSE value from the closure function
            mse = closure()
            
            epoch_errors_train.append(mse.item())
            
            # Create a plus or minus sign for the training error
            train_sign = "(-)"
            if epoch_errors_train[-1] < best_train:
                train_sign = "(+)"
                best_train = epoch_errors_train[-1]

            print(f"MSE: {train_sign}{mse} \n Iteration: {i}")
            mse_inference.append(float(mse))
            
            print(f"BC: {model.BC}")
            left_BC.append(float(model.BC[0,0]))
            right_BC.append(float(model.BC[1,0]))
            
            print(f"\n {model.BC.grad}")
            left_BC_grad.append(float(model.BC.grad[0,0]))
            right_BC_grad.append(float(model.BC.grad[1,0]))
            
            
        # Cut the used values from the data
        u = u[data_infer:,:]
        t = t[data_infer:]
        

    '''Evaluation'''
        
    model.eval()
    
    print(f"BC_final: {model.BC}")
    print(f"D_final: {model.D}")
    
    # Forward data through the model
    u_hat = model(t=t, u=u).detach().cpu()
    u = u.cpu()
    t = t.cpu()
    print(u.shape)
    print(model.BC)

    # Compute error
    mse = criterion(u_hat, u).item()
    print(f"MSE_final: {mse}")
    
    #
    # Visualize the data
    if config.data.type == "burger" and visualize:
        if infer_BC:
            # Plot the convergence of BC's and their gradients over iterations
            fig, ax = plt.subplots(1, 2, figsize=(18, 7), sharex=True)
            
            ax[0].plot(left_BC, label='left BC', color="red")
            ax[0].plot(right_BC, label='right BC', color="darkblue")
            ax[0].legend(loc="best", fontsize=18)
            ax[0].set_title("Convergence of the BC's", size=22)
            ax[0].axhline(y=4.0, color='saddlebrown')
            ax[0].axhline(y=-4.0, color='saddlebrown')
            ax[0].grid(True)
            
            ax[1].plot(left_BC_grad, label='left BC gradient', color="red")
            ax[1].plot(right_BC_grad, label='right BC gradient', color="darkblue")
            ax[1].legend(loc="best", fontsize=18)
            ax[1].set_title("Convergence of the Gradients of the BC's", size=22)
            ax[1].grid(True)
            
            plt.tight_layout()
            plt.draw()
            #plt.show()
            plt.savefig(f"{config.model.name}.pdf")
            
            # Plot the convergence of the error during inference
            fig, ax = plt.subplots(1, figsize=(9, 7))
            
            ax.plot(mse_inference, label='MSE During Inference', color="red")
            ax.legend(loc="best", fontsize=18)
            ax.set_title("Convergence of the Error", size=22)
            ax.grid(True)
            
            plt.tight_layout()
            plt.draw()
            #plt.show()
            plt.savefig(f"{config.model.name}_error.pdf")
            
            
        u_hat = np.transpose(u_hat)
        u = np.transpose(u)
     
        #=======================================================================
        # # plot the function learner
        # input_plot = th.linspace(-2, 2, 1000).unsqueeze(-1)
        # output_plot = model.func_nn(input_plot)
        # plt.plot(input_plot, output_plot.detach())
        #=======================================================================
         
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
     
        # u(t, x) over space
        h = ax[0].imshow(u, interpolation='nearest', 
                      extent=[t.min(), t.max(),
                              x.min(), x.max()],
                      origin='upper', aspect='auto')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
     
        ax[0].set_xlim(0, t.max())
        ax[0].set_ylim(x.min(), x.max())
        ax[0].legend(loc="upper right")
        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('$x$')
        ax[0].set_title('$u(t,x)$', fontsize = 10)
         
        h = ax[1].imshow(u_hat, interpolation='nearest', 
                      extent=[t.min(), t.max(),
                              x.min(), x.max()],
                      origin='upper', aspect='auto')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
     
        ax[1].set_xlim(0, t.max())
        ax[1].set_ylim(x.min(), x.max())
        ax[1].legend(loc="upper right")
        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel('$x$')
        ax[1].set_title('$u(t,x)$', fontsize = 10)
         
        # u(t, x) over time
        fig, ax = plt.subplots()
        line1, = ax.plot(x, u[:, 0], 'b-', linewidth=2, label='Exact')
        line2, = ax.plot(x, u_hat[:, 0], 'ro', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')    
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([-6.1, 6.1])
         
        anim = animation.FuncAnimation(fig,
                                       animate_1d,
                                       frames=len(t),
                                       fargs=(line1, line2, u, u_hat),
                                       interval=20)
    
        plt.tight_layout()
        plt.draw()
        plt.show()
        
    elif config.data.type == "allen_cahn" and visualize:
        if infer_BC:
            # Plot the convergence of BC's and their gradients over iterations
            fig, ax = plt.subplots(1, 2, figsize=(18, 7), sharex=True)
            
            ax[0].plot(left_BC, label='left BC', color="red")
            ax[0].plot(right_BC, label='right BC', color="darkblue")
            ax[0].legend(loc="best", fontsize=18)
            ax[0].set_title("Convergence of the BC's", size=22)
            ax[0].axhline(y=-1.5, color='saddlebrown')
            ax[0].axhline(y=1.5, color='saddlebrown')
            ax[0].grid(True)
            
            ax[1].plot(left_BC_grad, label='left BC gradient', color="red")
            ax[1].plot(right_BC_grad, label='right BC gradient', color="darkblue")
            ax[1].legend(loc="best", fontsize=18)
            ax[1].set_title("Convergence of the Gradients of the BC's", size=22)
            ax[1].grid(True)
            
            plt.tight_layout()
            plt.draw()
            #plt.show()
            plt.savefig(f"{config.model.name}.pdf")
            
            # Plot the convergence of the error during inference
            fig, ax = plt.subplots(1, figsize=(9, 7))
            
            ax.plot(mse_inference, label='MSE During Inference', color="red")
            ax.legend(loc="best", fontsize=18)
            ax.set_title("Convergence of the Error", size=22)
            ax.grid(True)
            
            plt.tight_layout()
            plt.draw()
            #plt.show()
            plt.savefig(f"{config.model.name}_error.pdf")
        
        u_hat = np.transpose(u_hat)
        u = np.transpose(u)
     
        #=======================================================================
        # # Plot the function learner
        # input_plot = th.linspace(-1, 1, 1000).unsqueeze(-1)
        # output_plot = model.func_nn(input_plot)
        # plt.plot(input_plot, output_plot.detach())
        # 
        # # Plot the actual function
        # plt.plot(input_plot, 5*(input_plot - input_plot**3))
        #=======================================================================
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
             
        # u(t, x) over space
        h = ax[0].imshow(u, interpolation='nearest', 
                      extent=[t.min(), t.max(),
                              x.min(), x.max()],
                      origin='upper', aspect='auto')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
     
        ax[0].set_xlim(0, t.max())
        ax[0].set_ylim(x.min(), x.max())
        ax[0].legend(loc="upper right")
        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('$x$')
        ax[0].set_title('$u(t,x)$', fontsize = 10)
         
        h = ax[1].imshow(u_hat, interpolation='nearest', 
                      extent=[t.min(), t.max(),
                              x.min(), x.max()],
                      origin='upper', aspect='auto')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
     
        ax[1].set_xlim(0, t.max())
        ax[1].set_ylim(x.min(), x.max())
        ax[1].legend(loc="upper right")
        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel('$x$')
        ax[1].set_title('$u(t,x)$', fontsize = 10)
         
        # u(t, x) over time
        fig, ax = plt.subplots()
        line1, = ax.plot(x, u[:, -1], 'b-', linewidth=2, label='Exact')
        line2, = ax.plot(x, u_hat[:, -1], 'ro', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')    
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([-1.1, 1.1])
     
        anim = animation.FuncAnimation(fig,
                                        animate_1d,
                                        frames=len(t),
                                        fargs=(line1, line2, u, u_hat),
                                        interval=20)
        plt.tight_layout()
        plt.draw()
        plt.show()
    
    return mse, model.BC


def animate_1d(t, axis1, axis2, field, field_hat):
    """
    Data animation function animating an image over time.
    :param t: The current time step
    :param axis: The matplotlib image object
    :param field: The data field
    :return: The matplotlib image object updated with the current time step's
        image date
    """
    axis1.set_ydata(field[:, t])
    axis2.set_ydata(field_hat[:, t])


def animate_2d(t, im1, im2, u_hat, u):
    """
    Data animation function animating an image over time.
    :param t: The current time step
    :param axis: The matplotlib image object
    :param field: The data field
    :return: The matplotlib image object updated with the current time step's
        image date
    """
    im1.set_array(u_hat[t,:,:].squeeze().t().detach())
    im2.set_array(u[t,:,:].squeeze().t().detach())



if __name__ == "__main__":
    th.set_num_threads(1)
    
    mse, inferred_BC = run_testing()

    print("Done.")