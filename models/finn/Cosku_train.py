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

sys.path.append("..")
from utils.configuration import Configuration
import utils.helper_functions as helpers
from finn import FINN_Burger, FINN_DiffSorp, FINN_DiffReact, FINN_AllenCahn


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
        #create a for loop and append the data or load a new data set for each batch in the training loop
        for idx in range(config.training.num_batches):
            if idx == 0:
                u = th.tensor(np.load(os.path.join(data_path, f"sample{idx}.npy")),
                                     dtype=th.float).to(device=device) #cut 5:45 check for the dimensions
                print(u.shape)
            else:
                u = th.cat((u, th.tensor(np.load(os.path.join(data_path, f"sample{idx}.npy")),
                                         dtype=th.float).to(device=device)))
                print(u.shape)
        
        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)

        dx = x[1]-x[0]
    
        # Initialize and set up the model
        model = FINN_Burger(
            u = u,
            D = np.array([1.0]),
            BC = np.array([[2.0],[-2.0]]),
            dx = dx,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="train",
            learn_coeff=True,
            learn_BC=True,
        ).to(device=device)
        
    
    elif config.data.type == "diffusion_sorption":
        # Load samples, together with x, y, and t series
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
        x = np.load(os.path.join(data_path, "x_series.npy"))
        sample_c = th.tensor(np.load(os.path.join(data_path, "sample_c.npy")),
                             dtype=th.float).to(device=device)
        sample_ct = th.tensor(np.load(os.path.join(data_path, "sample_ct.npy")),
                             dtype=th.float).to(device=device)
        
        dx = x[1]-x[0]
        u = th.stack((sample_c, sample_ct), dim=len(sample_c.shape))
        
        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)
        
        # Initialize and set up the model
        model = FINN_DiffSorp(
            u = u,
            D = np.array([0.5, 0.1]),
            BC = np.array([[1.0, 1.0], [0.0, 0.0]]),
            dx = dx,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="train",
            learn_coeff=True
        ).to(device=device)
    
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
            D = np.array([1.0]), # might set 2 or 3 if bc_learn=True
            BC = np.array([[-1.0], [-1.0]]),
            dx = dx,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="train",
            learn_coeff=True
        ).to(device=device)

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
    optimizer = th.optim.Adam(model.parameters(),
                                lr=config.training.learning_rate)

    #criterion = nn.MSELoss(reduction="mean") ## This was used before
    criterion = nn.MSELoss()

    #
    # Set up lists to save and store the epoch errors
    epoch_errors_train = []
    best_train = np.infty
    
    """
    TRAINING
    """

    a = time.time()

    #
    # Start the training and iterate over all epochs
    for epoch in range(config.training.epochs):
        
        epoch_start_time = time.time()
        
        # List to store the errors for each batch
        batch_errors = []
        
        # set the batch size for mini-batch training
        batch_size = config.training.batch_size
        print(f"len(u) / batch_size --> {len(u) / batch_size}")
        print(f"len(u) --> {len(u)}")
        
        for i in np.arange(len(u) / batch_size):
            print(f"i --> {i}")

            # Set the model to train mode
            model.train()
                
            # Reset the optimizer to clear data from previous iterations
            optimizer.zero_grad()
            
            left = int(i * batch_size)
            right = int((i+1) * batch_size)
            print(f"left --> {left}")
            print(f"right --> {right}")

            # Forward propagate and calculate loss function
            u_hat = model(t=t, u=u[left : right, :])

            mse = criterion(u_hat, u[left : right, :])
            
            mse.backward()
            
            #print(mse.grad)
            
            print(mse.item())
            print(f"D: {model.D}")
            print(f"BC: {model.BC}")
            print(f"BC.grad: {model.BC.grad}")
                
            optimizer.step()
            batch_errors.append(mse.item())
            
        epoch_errors_train.append(np.mean(batch_errors))

        # Create a plus or minus sign for the training error
        train_sign = "(-)"
        if epoch_errors_train[-1] < best_train:
            train_sign = "(+)"
            best_train = epoch_errors_train[-1]
            # Save the model to file (if desired)
            if config.training.save_model:
                # Start a separate thread to save the model
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
    

if __name__ == "__main__":
    th.set_num_threads(1)
    run_training(print_progress=True)

    print("Done.")
    
    
    
    

#===============================================================================
#     #
#     # Start the training and iterate over all epochs
#     for epoch in range(config.training.epochs):
#         
#         epoch_start_time = time.time()
#         
#         # set the batch size for mini-batch training
#         batch_size = config.training.batch_size
#         print(f"len(t) / batch_size --> {len(t) / batch_size}")
#         print(f"len(t) --> {len(t)}")
#         for i in np.arange(len(t) / batch_size):
#             print(f"i --> {i}")
#             
#             # Define the closure function that consists of resetting the
#             # gradient buffer, loss function calculation, and backpropagation
#             # It is necessary for LBFGS optimizer, because it requires multiple
#             # function evaluations
#             def closure():
#                 # Set the model to train mode
#                 model.train()
#                     
#                 # Reset the optimizer to clear data from previous iterations
#                 optimizer.zero_grad()
#                 
#                 left = int(i * batch_size)
#                 right = int((i+1) * batch_size)
#                 print(f"left --> {left}")
#                 print(f"right --> {right}")
#     
#                 # Forward propagate and calculate loss function
#                 u_hat = model(t=t[left : right], u=u[left : right, :])
#     
#                 mse = criterion(u_hat, u[left : right, :])
#                 
#                 mse.backward()
#                 
#                 #print(mse.grad)
#                 
#                 print(mse.item())
#                 print(f"D: {model.D}")
#                 print(f"BC: {model.BC}")
#                     
#                 return mse
#             
#             optimizer.step(closure)
#                 
#         # Extract the MSE value from the closure function
#         mse = closure()
#         
#         epoch_errors_train.append(mse.item())
# 
#         # Create a plus or minus sign for the training error
#         train_sign = "(-)"
#         if epoch_errors_train[-1] < best_train:
#             train_sign = "(+)"
#             best_train = epoch_errors_train[-1]
#             # Save the model to file (if desired)
#             if config.training.save_model:
#                 # Start a separate thread to save the model
#                 thread = Thread(target=helpers.save_model_to_file(
#                     model_src_path=os.path.abspath(""),
#                     config=config,
#                     epoch=epoch,
#                     epoch_errors_train=epoch_errors_train,
#                     epoch_errors_valid=epoch_errors_train,
#                     net=model))
#                 thread.start()
#===============================================================================