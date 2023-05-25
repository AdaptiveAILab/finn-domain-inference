#! env/bin/python3

"""
Main file for training a model with FINN
"""

import numpy as np
import torch as th
import torch.nn as nn
import os
import time
from threading import Thread
import sys
import matplotlib.pyplot as plt
import matplotlib

sys.path.append("..")
from utils.configuration import Configuration
import utils.helper_functions as helpers
from finn import FINN_Burger, FINN_AllenCahn, FINN_DiffReact, FINN_ShallowWater


def run_training(print_progress=True, model_number=None):

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
    # device = helpers.determine_device()
    device = th.device(config.general.device)
    
    if config.data.type == "burger":
        # Load samples, together with x, y, and t series
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
        x = np.load(os.path.join(data_path, "x_series.npy"))
        
        u = th.tensor(np.load(os.path.join(data_path, f"sample.npy")),
                      dtype=th.float).to(device=device)
        
        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)

        dx = x[1]-x[0]
    
        # Initialize and set up the model
        model = FINN_Burger(
            u = u,
            D = np.array([0.01/np.pi/dx**2]),
            BC = np.array([[0.0],[0.0]]),
            dx = dx,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="train",
            learn_coeff=True,
            learn_BC=False,
            train_mini_batch=False
        ).to(device=device)
        
        if model.learn_BC:
            # Cut the data before training
            data_cut = 30
            u = u[:data_cut,:]
            t = t[:data_cut]
            
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
            D = np.array([0.005/dx**2]), # might set 2 or 3 if bc_learn=True
            BC = np.array([[0.0], [0.0]]),
            dx = dx,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="train",
            learn_coeff=False,
            learn_BC=True,
            train_mini_batch=False
        ).to(device=device)
        
        if model.learn_BC:
            # Cut the data before training
            data_cut = 30
            u = u[:data_cut,:]
            t = t[:data_cut]
    
    elif config.data.type == "diffusion_reaction":
        # Load samples, together with x, y, and t series
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
        x = np.load(os.path.join(data_path, "x_series.npy"))
        y = np.load(os.path.join(data_path, "y_series.npy"))
        sample_u = th.tensor(np.load(os.path.join(data_path, "sample_u.npy")),
                             dtype=th.float).to(device=device)
        sample_v = th.tensor(np.load(os.path.join(data_path, "sample_v.npy")),
                             dtype=th.float).to(device=device)
        
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        
        
        u = th.stack((sample_u, sample_v), dim=len(sample_u.shape))
        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)
    
        # Initialize and set up the model
        model = FINN_DiffReact(
            u = u,
            D = np.array([5E-4/(dx**2), 1E-3/(dx**2)]),
            BC = np.zeros((4,2)),
            dx = dx,
            dy = dy,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="train",
            learn_coeff=True
        ).to(device=device)
    
    elif config.data.type == "shallow_water":
        import h5py
        
        # Load the data
        f = h5py.File(os.path.join(data_path, "wave_rotation.hdf5"), 'r')
        print(list(f.keys()))
        
        dset = f['train']['labels']
        
        # Transform the data into a tensor
        u = th.tensor(dset, dtype=th.float64).to(device=device)

        # Just take one sample from the data
        u = u[0]
        print(f"u -> {u.shape}")

        
        # ---------------- Set specific parameters depending on the data ---------------- #
        Lx = 1E+6                          # Length of domain in x-direction
        Ly = 1E+6                          # Length of domain in y-direction
        Nx = u.shape[1]                    # Number of grid points in x-direction
        Ny = u.shape[2]                    # Number of grid points in y-direction
        dx = Lx/(Nx - 1)                   # Grid spacing in x-direction
        dy = Ly/(Ny - 1)                   # Grid spacing in y-direction
        
        # g = 9.81             # Acceleration of gravity [m/s^2]
        # H = 10              # Depth of fluid [m] if two dimensional array, you can decide where shallow where deep
        
        # x = np.linspace(-Lx/2, Lx/2, Nx)  # Array with x-points
        # y = np.linspace(-Ly/2, Ly/2, Ny)  # Array with y-points
        
        # Generate t in order to make the data compatible for FINN
        dt = min(dx,dy)/300   # Time step (fulfills the CFL condition)
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
            mode="train",
            learn_coeff=False,
            learn_stencil=False,
            bias=False
        ).to(device=device)

        # Initialize all the weights as small values
        def weights_init(model):
            if isinstance(model, nn.Linear):
                th.nn.init.normal_(model.weight, 0.0, 0.01)

        model.apply(weights_init)

        # # Initialize weights of a particular network
        # for param in model.func_nn3.parameters():
        #     param.data = th.normal(th.ones(param.data.size()), 0.01).double()

        # # Look at the weights
        # for param in model.func_nn3.parameters():
        #     print(param.data.shape)
        #     print(param.data)

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    
    if print_progress:
        print("Trainable model parameters:", pytorch_total_params)

    # If desired, restore the network by loading the weights saved in the .pt file
    if config.training.continue_training:
        if print_progress: 
            print('Restoring model (that is the network\'s weights) from file...')
        model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                      "checkpoints",
                                      config.model.name,
                                      config.model.name + ".pt")))
        model.train()

    #
    # Set up an optimizer and the criterion (loss)
    optimizer = th.optim.Adam(model.parameters(),
                                lr=config.training.learning_rate)

    criterion = nn.MSELoss(reduction="mean")

    #
    # Set up lists to save and store the epoch errors
    epoch_errors_train = []
    best_train = np.infty
    
    """
    TRAINING
    """

    if model.train_mini_batch: # mini batch training for SWE not implemented
        
        # Load boundary conditions of the mini-batches
        left_BC = th.tensor(np.load(os.path.join(data_path, "left_BC.npy")),
                                         dtype=th.float).to(device=device)
        right_BC = th.tensor(np.load(os.path.join(data_path, "right_BC.npy")),
                                         dtype=th.float).to(device=device)
        
        print(left_BC)
        print(right_BC)
                                         
        # Load t series for mini-batch training
        t = th.tensor(np.load(os.path.join(data_path, "t_series_mini_batches.npy")),
                      dtype=th.float).to(device=device)
        
        shuffle = np.random.permutation(config.training.batch_size)
        print(shuffle)
        
        for idx in np.arange(config.training.batch_size):
              
            data = th.tensor(np.load(os.path.join(data_path, f"sample_{str(shuffle[idx]).zfill(2)}.npy")),
                                         dtype=th.float).to(device=device)
              
            if idx == 0:
                u = data.unsqueeze(2)
                  
                BC = th.tensor([[left_BC[shuffle[idx]]], [right_BC[shuffle[idx]]]]).unsqueeze(0)
            else:
                u = th.cat((u, data.unsqueeze(2)), dim=2)
                BC = th.cat((BC, th.tensor([[left_BC[shuffle[idx]]], [right_BC[shuffle[idx]]]]).unsqueeze(0)), dim=0)
                 
        model.BC = BC
        print(model.BC)
        print(model.BC.shape)
        print(u.shape)
        
    
    if model.learn_BC:
        # Initialize lists to store
        left_BC = []
        right_BC = []
        
        left_BC_grad = []
        right_BC_grad = []
        
        mse_train = []
        
    a = time.time()


    online_learning = False
        
    if online_learning:
        
        online_errors = np.zeros((config.training.epochs, len(t) - 1))
        #
        # Start the training and iterate over all epochs
        for epoch in range(config.training.epochs):
    
            epoch_start_time = time.time()

            if config.data.type == "shallow_water":
                u0_temp = initial_state
            else:
                u0_temp = u[0]
            j = -1
            
            # Set teacher forcing time steps
            teacher_forcing = 100
            
            for t0, t1, u_i, in zip(t[:-1], t[1:], u[1:]):
                t_i = th.tensor([t0, t1]).to(device=device)
                # Define the closure function that consists of resetting the
                # gradient buffer, loss function calculation, and backpropagation
                # It is necessary for LBFGS optimizer, because it requires multiple
                # function evaluations
                def closure():
                    # Set the model to train mode
                    model.train()
                        
                    # Reset the optimizer to clear data from previous iterations
                    optimizer.zero_grad()
        
                    # Set u_hat_i as global such that it is accessible outside of the function
                    global u_hat_i

                    # Forward propagate and calculate loss function
                    u_hat_i = model(t=t_i, u=u0_temp.unsqueeze(0))

                    if config.data.type == "shallow_water":
                        mse = criterion(u_hat_i[-1,:,:,0], u_i)
                    else:
                        mse = criterion(u_hat_i[-1], u_i)
                    
                    mse.backward()
                    
                    print(mse.item())
                    #print(model.D)
                    online_errors[epoch][j] = mse.item()
                    
                    return mse
                    
                optimizer.step(closure)

                # Increment j
                j += 1
                
                if j < teacher_forcing:
                    # Apply teacher forcing
                    if config.data.type == "shallow_water":
                        u0_temp = th.cat((u[j+1].unsqueeze(-1), u_hat_i[-1,:,:,1:]),dim=-1).detach()
                    else:
                        u0_temp = u[j+1].detach()
                else:
                    # Train in close loop
                    u0_temp = u_hat_i[-1].detach()

            # Extract the MSE value from the closure function
            mse = closure()
                
            #===================================================================
            # # Get the last online MSE as the error of the epoch 
            # epoch_errors_train.append(mse.item())
            #===================================================================
            
            # Get the averaged MSE as the error of the epoch
            epoch_errors_train.append(np.mean(online_errors[epoch]))
    
            # Create a plus or minus sign for the training error
            train_sign = "(-)"
            if epoch_errors_train[-1] < best_train:
                train_sign = "(+)"
                best_train = epoch_errors_train[-1]
                # Save the model to file (if desired)
                if config.training.save_model:
                    #Start a separate thread to save the model
                    thread = Thread(target=helpers.save_model_to_file(
                        model_src_path=os.path.abspath(""),
                        config=config,
                        epoch=epoch,
                        epoch_errors_train=epoch_errors_train,
                        epoch_errors_valid=epoch_errors_train,
                        net=model))
                    thread.start()
    
            
            #
            # Print progress to the console
            if print_progress:
                print(f"Epoch {str(epoch+1).zfill(int(np.log10(config.training.epochs))+1)}/{str(config.training.epochs)} took {str(np.round(time.time() - epoch_start_time, 2)).ljust(5, '0')} seconds. \t\tAverage epoch training error: {train_sign}{str(np.round(epoch_errors_train[-1], 10)).ljust(12, ' ')}")
    else: ### Turns on the batch learning (no online learning anymore)
        #
        # Start the training and iterate over all epochs

        for epoch in range(config.training.epochs):
            
            epoch_start_time = time.time()
            
            # Define the closure function that consists of resetting the
            # gradient buffer, loss function calculation, and backpropagation
            # It is necessary for LBFGS optimizer, because it requires multiple
            # function evaluations
            def closure():
                # Set the model to train mode
                model.train()
                    
                # Reset the optimizer to clear data from previous iterations
                optimizer.zero_grad()

                if config.data.type == "shallow_water":
                    # Forward propagate and calculate loss function
                    u_hat = model(t=t, u=initial_state.unsqueeze(0))
                    mse = criterion(u_hat[...,0], u)
                    # mse = criterion(u_hat, u) #[32, 32, 256]
                else:
                    # Forward propagate and calculate loss function
                    u_hat = model(t=t, u=u)
                    mse = criterion(u_hat, u)

                mse.backward()

                # print(mse.item())
                # print(model.D)
                # print(model.g)
                # print(f'H --> {model.H}')

                if model.learn_BC:
                    print(f"BC: {model.BC}")
                    left_BC.append(float(model.BC[0,0]))
                    right_BC.append(float(model.BC[1,0]))
                    
                    print(f"\n BC_grad: {model.BC.grad}")
                    left_BC_grad.append(float(model.BC.grad[0,0]))
                    right_BC_grad.append(float(model.BC.grad[1,0]))
                    
                    print(f"mse_train: {mse.item()}")
                    mse_train.append(float(mse))
                    
                return mse
                
            optimizer.step(closure)

            # Extract bias
            # print(model.func_nn2[-1].bias.item())
                
            # Extract the MSE value from the closure function
            mse = closure()
            epoch_errors_train.append(mse.item())
    
            # Create a plus or minus sign for the training error
            train_sign = "(-)"
            if epoch_errors_train[-1] < best_train:
                train_sign = "(+)"
                best_train = epoch_errors_train[-1]
                # Save the model to file (if desired)
                if config.training.save_model:
                    #Start a separate thread to save the model
                    thread = Thread(target=helpers.save_model_to_file(
                        model_src_path=os.path.abspath(""),
                        config=config,
                        epoch=epoch,
                        epoch_errors_train=epoch_errors_train,
                        epoch_errors_valid=epoch_errors_train,
                        net=model))
                    thread.start()
    
            
            #
            # Print progress to the console
            if print_progress:
                print(f"Epoch {str(epoch+1).zfill(int(np.log10(config.training.epochs))+1)}/{str(config.training.epochs)} took {str(np.round(time.time() - epoch_start_time, 2)).ljust(5, '0')} seconds. \t\tAverage epoch training error: {train_sign}{str(np.round(epoch_errors_train[-1], 28)).ljust(30, ' ')}")

    b = time.time()
    if print_progress:
        print('\nTraining took ' + str(np.round(b - a, 2)) + ' seconds.\n\n')
    
    # Plot the  averaged epoch errors
    fig, ax = plt.subplots(1, figsize=(9, 5))
     
    ax.plot(np.log(epoch_errors_train[2:]), label='Averaged Epoch Errors', color="red")
    #ax.legend(np.arange(1, config.training.epochs+1), loc="best", fontsize=16)
    #ax.set_title("Convergence of the Error", size=22)
    ax.grid(True, linewidth=0.5)
     
    plt.tight_layout()
    plt.draw()
    #plt.show()
    plt.savefig(f"averaged_epoch_errors.pdf")
    
    
    if online_learning:
        # Plot the errors in each epoch (last 15 Epochs)
        fig, ax = plt.subplots(1, figsize=(18, 9), sharex=True)
        
        # Plot the last 15 epochs
        ax.plot(np.transpose(online_errors[-15:,:]))
        ax.legend(np.arange(1, config.training.epochs+1), loc="best", fontsize=16)
        #ax.set_title("Convergence of the Error", size=22)
        ax.grid(True)
         
        plt.tight_layout()
        plt.draw()
        #plt.show()
        plt.savefig(f"online_errors_last15.pdf")
        
        
        # Plot the errors in each epoch (first 15 Epochs)
        fig, ax = plt.subplots(1, figsize=(18, 9), sharex=True)
        
        # Plot the last 15 epochs
        ax.plot(np.transpose(online_errors[:15,:]))
        ax.legend(np.arange(1, config.training.epochs+1), loc="best", fontsize=16)
        #ax.set_title("Convergence of the Error", size=22)
        ax.grid(True)
         
        plt.tight_layout()
        plt.draw()
        #plt.show()
        plt.savefig(f"online_errors_first15.pdf")
        
        
    if model.learn_BC:
        matplotlib.rc('xtick', labelsize=16) 
        matplotlib.rc('ytick', labelsize=16) 
        # Plot the convergence of BC's and their gradients over epochs
        fig, ax = plt.subplots(1, 2, figsize=(18, 5), sharex=True)
                
        ax[0].plot(left_BC, label='left BC', color="red")
        ax[0].plot(right_BC, label='right BC', color="darkblue")
        ax[0].legend(loc="best", fontsize=20)
        ax[0].set_title("Convergence of the BC's", size=26)
        ax[0].set_xlabel('Iterations', size=22)
        ax[0].set_ylabel('BC Values', size=22)
        ax[0].axhline(y=1.0, color='saddlebrown')
        ax[0].axhline(y=-1.0, color='saddlebrown')
        ax[0].grid(True, linewidth=0.5)
        
        ax[1].plot(left_BC_grad, label='left BC gradient', color="red")
        ax[1].plot(right_BC_grad, label='right BC gradient', color="darkblue")
        ax[1].legend(loc="best", fontsize=20)
        ax[1].set_title("Convergence of the Gradients of the BC's", size=26)
        ax[1].set_xlabel('Iterations', size=22)
        ax[1].set_ylabel('Gradients', size=22)
        ax[1].axhline(y=0.0, color='saddlebrown')
        ax[1].grid(True, linewidth=0.5)
        
        #plt.xticks(fontsize=16)
        #plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.draw()
        #plt.show()
        plt.savefig(f"{config.model.name}.pdf")
        
        # Plot the convergence of the error during inference
        fig, ax = plt.subplots(1, figsize=(9, 7))
        
        ax.plot(mse_train, label='MSE During Training', color="red")
        ax.legend(loc="best", fontsize=18)
        ax.set_title("Convergence of the Error", size=22)
        ax.grid(True)
        
        plt.tight_layout()
        plt.draw()
        #plt.show()
        #plt.savefig(f"{config.model.name}_error.pdf")
    

if __name__ == "__main__":
    th.set_num_threads(1)
    run_training(print_progress=True)

    print("Done.")