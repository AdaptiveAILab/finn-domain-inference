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

sys.path.append("..")
from utils.configuration import Configuration
import utils.helper_functions as helpers
from finn import FINN_Burger, FINN_AllenCahn


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
        x = np.load(os.path.join(data_path, "x_series.npy"))## [5:45] # cut this maybe 5:45
        
        u = th.tensor(np.load(os.path.join(data_path, f"sample.npy")),
                      dtype=th.float).to(device=device) #cut 5:45 check for the dimensions
        
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
            learn_coeff=False,
            learn_BC=True,
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

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    
    if print_progress:
        print("Trainable model parameters:", pytorch_total_params)

    # If desired, restore the network by loading the weights saved in the .pt
    # file
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
    optimizer = th.optim.LBFGS(model.parameters(),
                                lr=config.training.learning_rate)

    #criterion = nn.MSELoss(reduction="mean") ## This was used before
    criterion = nn.MSELoss(reduction="mean")

    #
    # Set up lists to save and store the epoch errors
    epoch_errors_train = []
    best_train = np.infty
    
    """
    TRAINING
    """

    if model.train_mini_batch:
        
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

            # Forward propagate and calculate loss function
            
            u_hat = model(t=t, u=u)

            mse = criterion(u_hat, u)
            
            mse.backward()
            
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
            print(f"Epoch {str(epoch+1).zfill(int(np.log10(config.training.epochs))+1)}/{str(config.training.epochs)} took {str(np.round(time.time() - epoch_start_time, 2)).ljust(5, '0')} seconds. \t\tAverage epoch training error: {train_sign}{str(np.round(epoch_errors_train[-1], 10)).ljust(12, ' ')}")

    b = time.time()
    if print_progress:
        print('\nTraining took ' + str(np.round(b - a, 2)) + ' seconds.\n\n')
    
    if model.learn_BC:
        # Plot the convergence of BC's and their gradients over epochs
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
        
        ax.plot(mse_train, label='MSE During Training', color="red")
        ax.legend(loc="best", fontsize=18)
        ax.set_title("Convergence of the Error", size=22)
        ax.grid(True)
        
        plt.tight_layout()
        plt.draw()
        #plt.show()
        plt.savefig(f"{config.model.name}_error.pdf")
    

if __name__ == "__main__":
    th.set_num_threads(1)
    run_training(print_progress=True)

    print("Done.")