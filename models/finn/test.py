#! env/bin/python3

"""
Main file for testing (evaluating) a FINN model
"""

import numpy as np
import torch as th
import torch.nn as nn
import os
from threading import Thread
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import sys


sys.path.append("..")
from utils.configuration import Configuration
import utils.helper_functions as helpers
# from plot import show3d
from finn import FINN_Burger, FINN_AllenCahn, FINN_ShallowWater


def run_testing(visualize=True, model_number=None, infer_BC=False, infer_H=True):

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
            learn_coeff=True,
            learn_BC=False,
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
            learn_coeff=True,
            learn_BC=False,
            train_mini_batch=False
        ).to(device=device)
        
        
    elif config.data.type == "shallow_water":
        import h5py
        
        # Load the data
        f = h5py.File(os.path.join(data_path, "wave_rotation.hdf5"), 'r')
        print(list(f.keys()))
        
        dset = f['train']['labels']
        
        # Transform the data into a tensor
        u = th.tensor(dset, dtype=th.float).to(device=device)
        
        # Just take one sample from the whole data
        u = u[0].double()
        print(f"u -> {u.shape}")
        
        # ---------------- Set specific parameters depending on the data ---------------- #
        Lx = 1E+6                           # Length of domain in x-direction
        Ly = 1E+6                           # Length of domain in y-direction
        Nx = u.shape[1]                     # Number of grid points in x-direction
        Ny = u.shape[2]                     # Number of grid points in y-direction
        dx = Lx/(Nx - 1)                    # Grid spacing in x-direction
        dy = Ly/(Ny - 1)                    # Grid spacing in y-direction
        
        # g = 9.81             # Acceleration of gravity [m/s^2]
        # H = 100              # Depth of fluid [m] if two dimensional array, you can decide where shallow where deep
        
        x = np.linspace(-Lx/2, Lx/2, Nx)  # Array with x-points
        y = np.linspace(-Ly/2, Ly/2, Ny)  # Array with y-points
        
        # Generate t in order to make the data compatible for FINN
        th.set_printoptions(precision=10)
        dt = min(dx, dy) / 300  # Time step (fulfills the CFL condition)
        # dt = 0.1*min(dx, dy)/np.sqrt(g)   # Time step (defined from the CFL condition)
        sample_interval = 1
        t_steps = u.shape[0]

        t_temp = th.zeros(t_steps * sample_interval, dtype=th.float64, device=device)

        for i in range(1, t_steps * sample_interval):
            t_temp[i] = t_temp[i-1] + dt

        t = th.zeros(t_steps, dtype=th.float64, device=device)
        counter = 1

        for i in range(1, t_steps * sample_interval):
            if i % sample_interval == 0:
                t[counter] = t_temp[i]
                counter += 1


        # initialize velo_x and velo_y
        velo_x = th.zeros(Nx, Ny, device=device)
        velo_y = th.zeros(Nx, Ny, device=device)

        
        # Stack all the tensor to integrate all the equations
        initial_state = th.stack((u[0], velo_x, velo_y),dim=-1)
        
        
        # Initialize and set up the model
        model = FINN_ShallowWater(
            u = u,
            D = np.array([0.0]),
            BC = np.zeros((4,1)),
            dx = dx,
            dy = dy,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="test",
            learn_coeff=False,
            learn_stencil=False,
            bias=False
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
    
    
    '''  BC INFERENCE  '''
    
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
        data_infer = 15
        
        # Set the number of iteration
        iterations = 300
        
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

    '''  TOPOGRAPHY INFERENCE  '''

    # Block for inferring the topography
    if infer_H and config.data.type == "shallow_water":
        # Set all trainable parameters to false
        for p in model.parameters():
            p.requires_grad = False

        # Set H as learnable parameter
        model.H = nn.Parameter(model.H)

        # Count number of trainable parameters
        pytorch_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"Trainable model parameters for the spatial domain: {pytorch_total_params}\n")

        # If desired, restore the network weights after inference
        if config.inference.continue_inference:
            print("Restoring the inference model.")
            model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                                       "checkpoints",
                                                       'infer_' + config.model.name,
                                                       'infer_' + config.model.name + ".pt")))

        # Set up an optimizer
        optimizer = th.optim.Adam(model.parameters(), lr=config.training.learning_rate)

        #
        # Set up lists to save and store the epoch errors
        epoch_errors_infer = []
        best_infer = np.infty

        # Set the number of iteration
        iterations = 10

        # Initialize a list to store inferred H values
        if config.inference.save_model and not config.inference.continue_inference:
            H_progress = th.zeros((iterations, model.H.shape[0], model.H.shape[1]), dtype=th.float64, device=device)

        elif config.inference.save_model and config.inference.continue_inference:
            print("Restoring the previous progress of H.")
            H_progress_temp = th.tensor(np.load(os.path.join(os.path.abspath(""),
                                                             "checkpoints",
                                                             'infer_' + config.model.name,
                                                             "H_progress.npy")), dtype=th.float64, device=device)
            H_progress = th.cat((H_progress_temp,
                                 th.zeros((iterations, model.H.shape[0], model.H.shape[1]))), dim=0)


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
                u_hat = model(t=t, u=initial_state.unsqueeze(0))

                mse = criterion(u_hat[...,0], u)

                mse.backward()

                return mse

            optimizer.step(closure)

            # Extract the MSE value from the closure function
            mse = closure()

            epoch_errors_infer.append(mse.item())

            # Create a plus or minus sign for the training error
            infer_sign = "(-)"
            if epoch_errors_infer[-1] < best_infer:
                infer_sign = "(+)"
                best_infer = epoch_errors_infer[-1]
                # Save the model to file (if desired)
                if config.inference.save_model:
                    # Start a separate thread to save the model
                    thread = Thread(target=helpers.save_model_to_file(
                        model_src_path=os.path.abspath(""),
                        config=config,
                        epoch=i,
                        epoch_errors_train=epoch_errors_infer,
                        epoch_errors_valid=epoch_errors_infer,
                        net=model,
                        infer=True))
                    thread.start()

            print(f"Iteration: {i}, MSE: {infer_sign}{mse}")

            # Save the H values after each iteration
            if config.inference.save_model and not config.inference.continue_inference:
                H_progress[i] = model.H
            elif config.inference.save_model and config.inference.continue_inference:
                H_progress[len(H_progress_temp) + i] = model.H

    '''Evaluation'''

    # Restore the model
    if infer_H:
        print('Restoring the model')
        model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                                   "checkpoints",
                                                   'infer_' + config.model.name,
                                                   'infer_' + config.model.name + ".pt")))
    model.eval()
    
    #print(f"BC_final: {model.BC}")
    #print(f"D_final: {model.D}")
    
    if config.data.type == "shallow_water":
        # Forward data through the model
        u_hat = model(t=t, u=initial_state.unsqueeze(0)).detach().cpu()
        # Compute error
        mse = criterion(u_hat[...,0], u)
        print(f"MSE_final: {mse}")
        print(f"Total Quantity u -> {th.sum(u)}")
        print(f"Total Quantity u_hat -> {th.sum(u_hat[...,0])}")
        print((th.sum(u) - th.sum(u_hat[...,0])).item())
        print((th.sum(u) == th.sum(u_hat[...,0])).item())
    else:
        # Forward data through the model
        u_hat = model(t=t, u=u).detach().cpu()
        # Compute error
        mse = criterion(u_hat, u).item()
        print(f"MSE_final: {mse}")

    # Look at the weights
    # for param in model.parameters():
    #     print(param.data.shape)
    #     print(param.data)

    th.set_printoptions(profile='full')
    print(f"Inferred H: \n f{model.H}")
    
    u = u.cpu()
    t = t.cpu()

    #
    # Visualize the data
    if visualize and config.data.type == "burger" or config.data.type == "allen_cahn":
        if infer_BC:
            matplotlib.rc('xtick', labelsize=16)
            matplotlib.rc('ytick', labelsize=16)
            # Plot the convergence of BC's and their gradients over epochs
            fig, ax = plt.subplots(1, 2, figsize=(18, 6), sharex=True)

            ax[0].plot(left_BC, label='left BC', color="red")
            ax[0].plot(right_BC, label='right BC', color="darkblue")
            ax[0].legend(loc="best", fontsize=20)
            ax[0].set_title("Convergence of the BC's", size=26)
            ax[0].set_xlabel('Iterations', size=22)
            ax[0].set_ylabel('BC Values', size=22)
            ax[0].axhline(y=1.5, color='saddlebrown')
            ax[0].axhline(y=-1.5, color='saddlebrown')
            ax[0].grid(True, linewidth=0.5)

            cut_grad_arr = 800
            ax[1].plot(np.arange(cut_grad_arr, len(left_BC_grad)), left_BC_grad[cut_grad_arr:], label='left BC gradient', color="red")
            ax[1].plot(np.arange(cut_grad_arr, len(right_BC_grad)), right_BC_grad[cut_grad_arr:], label='right BC gradient', color="darkblue")
            ax[1].legend(loc="best", fontsize=20)
            ax[1].set_title("Convergence of the Gradients of the BC's", size=26)
            ax[1].set_xlabel('Iterations', size=22)
            ax[1].set_ylabel('Gradients', size=22)
            ax[1].axhline(y=0.0, color='saddlebrown')
            ax[1].grid(True, linewidth=0.5)

            plt.tight_layout()
            plt.draw()
            # plt.show()
            plt.savefig(f"{config.model.name}.pdf")

            # Plot the convergence of the error during inference
            fig, ax = plt.subplots(1, figsize=(13, 7))

            ax.plot(np.log(mse_inference), label='MSE During Inference', color="red")
            ax.legend(loc="best", fontsize=18)
            ax.set_title("Convergence of the Error", size=26)
            ax.set_xlabel('Iterations', size=22)
            ax.set_ylabel('Log-Error', size=22)

            ax.grid(True, linewidth=0.5)

            plt.tight_layout()
            plt.draw()
            # plt.show()
            plt.savefig(f"{config.model.name}_error.pdf")

        u_hat = np.transpose(u_hat)
        u = np.transpose(u)

        # =======================================================================
        # # plot the function learner
        # input_plot = th.linspace(-2, 2, 1000).unsqueeze(-1)
        # output_plot = model.func_nn(input_plot)
        # plt.plot(input_plot, output_plot.detach())
        # =======================================================================

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        plt.style.use('ggplot')

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
        ax[0].set_title('$u(t,x)$', fontsize=10)

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
        ax[1].set_title('$u(t,x)$', fontsize=10)

        # u(t, x) over time
        fig, ax = plt.subplots()
        line1, = ax.plot(x, u[:, 0], 'ro', linewidth=2, label='Exact')
        line2, = ax.plot(x, u_hat[:, 0], 'b-', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$', fontsize=24)
        ax.set_ylabel('$u(t,x)$', fontsize=24)
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([-1.6, 1.6])
        ax.legend(loc="upper left", fontsize=16)
        ax.grid(True, linewidth=0.3)
        if cut_data:
            ax.axvline(x=x[cut_data_size], color='saddlebrown', linewidth=2.5)
            ax.axvline(x=x[-cut_data_size], color='saddlebrown', linewidth=2.5)

        anim = animation.FuncAnimation(fig,
                                       animate_1d,
                                       frames=len(t) - 1,
                                       fargs=(line1, line2, u, u_hat),
                                       interval=20)

        plt.tight_layout()
        plt.draw()
        # plt.show()

        f = f"{config.model.name}_animation.mp4"
        writervideo = animation.FFMpegWriter(fps=60)
        anim.save(f, writer=writervideo)
     
    elif config.data.type == "shallow_water" and visualize:
        timesteps = len(t)

        # Plot the wave activity at one position
        fig, ax = plt.subplots(1, 2, figsize=[8, 2])

        ax[0].plot(range(len(u)), u[:, 17, 17])
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Wave amplitude")
        ax[0].set_title("Ground Truth")
        ax[0].set_xlim([0, timesteps - 1])

        ax[1].plot(range(len(u_hat)), u_hat[:, 17, 17, 0])
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Wave amplitude")
        ax[1].set_title("Prediction")
        ax[1].set_xlim([0, timesteps - 1])

        plt.tight_layout()
        plt.show()
        plt.close()

        # Animate the spatio-temporal wave
        fig, ax = plt.subplots(1, 2, figsize=(6, 6))

        im1 = ax[0].imshow(u_hat[0, :, :, 0], interpolation='nearest', vmin=-0.6, vmax=0.6, cmap="Blues")
        ax[0].set_title("Prediction")


        im2 = ax[1].imshow(u[0, :, :], interpolation='nearest', vmin=-0.6, vmax=0.6, cmap="Blues")
        ax[1].set_title("Ground Truth")

        anim = animation.FuncAnimation(fig,
                                       animate_2d,
                                       frames=timesteps,
                                       fargs=(im1, im2, u_hat[..., 0], u),
                                       interval=10)
        plt.tight_layout()
        plt.draw()
        plt.show()
        plt.close()

        # # Save the animation
        # f = f"{config.model.name}_animation.mp4"
        # writervideo = animation.FFMpegWriter(fps=60)
        # anim.save(f, writer=writervideo)

        if infer_H and config.inference.save_model:
            # Convert H_progress into a numpy-array
            H_progress = H_progress.detach().cpu().numpy()
            print(H_progress.shape)

            # Save H_progress into a npy-file
            name = "H_progress.npy"
            data_path = os.path.join(os.path.abspath(""), "checkpoints", 'infer_' + config.model.name)
            np.save(file=os.path.join(data_path, name), arr=H_progress)

            # Animate the inferred topography
            timesteps = len(H_progress)
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

            im = ax.imshow(H_progress[0, :, :], interpolation='nearest', cmap="Blues")
            ax.set_title("Inferred Topography")

            anim = animation.FuncAnimation(fig,
                                           animate,
                                           frames=timesteps,
                                           fargs=(im, H_progress),
                                           interval=10)
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.close()
            
    return mse, model.BC

def animate_3d(t, im, field):
    """
    Data animation function animating an image over time.
    :param t: The current time step
    :param im: The matplotlib image object
    :param field: The data field
    :return: The matplotlib image object updated with the current time step's
        image date
    """
    im.set_array(field[t, :, :, :])
    return im



def animate(t, im, field):
    """
    Data animation function animating an image over time.
    :param t: The current time step
    :param im: The matplotlib image object
    :param field: The data field
    :return: The matplotlib image object updated with the current time step's
        image date
    """
    im.set_array(field[t, :, :])
    return im

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