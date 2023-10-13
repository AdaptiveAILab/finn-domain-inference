#! env/bin/python3

"""
Main file for training a model
"""

import numpy as np
import torch as th
import torch.nn as nn
import os
import time
from threading import Thread
import random
import sys
import matplotlib.pyplot as plt

sys.path.append("..")
from utils.configuration import Configuration
import utils.helper_functions as helpers
from phydnet import ConvLSTM, PhyCell, EncoderRNN
from constrain_moments import K2M


def run_training(print_progress=True, model_number=None):
    
    #decides if mini batch training or not
    train_multi_batch = False

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
    
    if config.model.small:
        input_dim = 32
        hidden_dims = [32]
        n_layers_convcell = 1
    else:
        input_dim = 64
        hidden_dims = [128,128,64]
        n_layers_convcell = 3

    if config.data.type == "burger":
        
        if train_multi_batch:
            left_BC = th.tensor(np.load(os.path.join(data_path, "left_BC.npy")),
                                         dtype=th.float).to(device=device)
            right_BC = th.tensor(np.load(os.path.join(data_path, "right_BC.npy")),
                                             dtype=th.float).to(device=device)
                                             
            print(left_BC)
            print(right_BC)
            
            shuffle = np.random.permutation(config.training.batch_size)
            print(shuffle)
            
            for idx in np.arange(config.training.batch_size):
                u = th.tensor(np.load(os.path.join(data_path, f"sample_{str(shuffle[idx]).zfill(2)}.npy")),
                                                 dtype=th.float).to(device=device)
                if idx == 0:
                    # input_tensor : torch.Size([batch_size, input_length, channels, cols, rows])
                    sample_length = u.shape[0]
                    input_tensor = u[:sample_length//2].unsqueeze(0).unsqueeze(-2).to(device=device)
                    target_tensor = u[sample_length//2:].unsqueeze(0).unsqueeze(-2).to(device=device)
                    
                    bc = th.tensor([[[left_BC[shuffle[idx]], right_BC[shuffle[idx]]]]])
                else:
                    sample_length = u.shape[0]
                    input_tensor = th.cat((input_tensor, u[:sample_length//2].unsqueeze(0).unsqueeze(-2)),dim=0).to(device=device)
                    target_tensor = th.cat((target_tensor, u[sample_length//2:].unsqueeze(0).unsqueeze(-2)),dim=0).to(device=device)
                    
                    bc = th.cat((bc, th.tensor([[[left_BC[shuffle[idx]], right_BC[shuffle[idx]]]]])), dim=0)
                print(input_tensor.shape)
                print(target_tensor.shape)
        else:
            # Load samples
            u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                                 dtype=th.float).to(device=device)

            # Add noise to the data
            u = u + th.normal(th.zeros_like(u),th.ones_like(u)*config.data.noise)

            # Uncomment if you want to train a shorter sequence
            # u = u[:30]
            # print(u.shape)

            bc = th.tensor([[[0.5, -0.5]]]).to(device)

            # input_tensor : torch.Size([batch_size, input_length, channels, cols, rows])
            sample_length = u.shape[0]
            input_tensor = u[:sample_length//2].unsqueeze(0).unsqueeze(-2).to(device=device)
            target_tensor = u[sample_length//2:].unsqueeze(0).unsqueeze(-2).to(device=device)
        
        # Initialize and set up the model
        phycell  =  PhyCell(input_shape=(input_tensor.shape[-1]//4+1),
                            input_dim=input_dim,
                            F_hidden_dims=[7],
                            n_layers=1,
                            kernel_size=7,
                            device=device) 
        
        convcell =  ConvLSTM(input_shape=(input_tensor.shape[-1]//4+1),
                             input_dim=input_dim,
                             hidden_dims=hidden_dims,
                             n_layers=n_layers_convcell,
                             kernel_size=3,
                             device=device)
        
        model  = EncoderRNN(phycell,
                              convcell,
                              input_channels=1,
                              input_dim=(input_tensor.shape[-1],),
                              _1d=True,
                              bc=bc,
                              learn_BC=False,
                              device=device,
                              sigmoid=False,
                              small=config.model.small)
        
        constraints = th.zeros((7,7)).to(device)
        ind = 0
        for i in range(0,7):
            constraints[ind,i] = 1
            ind +=1
        
                
    elif config.data.type == "allen_cahn":
        
        if train_multi_batch:
            left_BC = th.tensor(np.load(os.path.join(data_path, "left_BC.npy")),
                                         dtype=th.float).to(device=device)
            right_BC = th.tensor(np.load(os.path.join(data_path, "right_BC.npy")),
                                             dtype=th.float).to(device=device)
                                             
            print(left_BC)
            print(right_BC)
            
            shuffle = np.random.permutation(config.training.batch_size)
            print(shuffle)
            
            for idx in np.arange(config.training.batch_size):
                u = th.tensor(np.load(os.path.join(data_path, f"sample_{str(shuffle[idx]).zfill(2)}.npy")),
                                                 dtype=th.float).to(device=device)
                if idx == 0:
                    # input_tensor : torch.Size([batch_size, input_length, channels, cols, rows])
                    sample_length = u.shape[0]
                    input_tensor = u[:sample_length//2].unsqueeze(0).unsqueeze(-2).to(device=device)
                    target_tensor = u[sample_length//2:].unsqueeze(0).unsqueeze(-2).to(device=device)
                    
                    bc = th.tensor([[[left_BC[shuffle[idx]], right_BC[shuffle[idx]]]]])
                else:
                    sample_length = u.shape[0]
                    input_tensor = th.cat((input_tensor, u[:sample_length//2].unsqueeze(0).unsqueeze(-2)),dim=0).to(device=device)
                    target_tensor = th.cat((target_tensor, u[sample_length//2:].unsqueeze(0).unsqueeze(-2)),dim=0).to(device=device)
                    bc = th.cat((bc, th.tensor([[[left_BC[shuffle[idx]], right_BC[shuffle[idx]]]]])), dim=0)
                print(input_tensor.shape)
                print(target_tensor.shape)
        else:
            # Load samples
            u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                                 dtype=th.float).to(device=device)

            # Add noise to the data
            u = u + th.normal(th.zeros_like(u),th.ones_like(u)*config.data.noise)

            # Uncomment if you want to train a shorter sequence
            # u = u[:30]
            # print(u.shape)

            bc = th.tensor([[[-0.5, 0.5]]]).to(device)

            # input_tensor : torch.Size([batch_size, input_length, channels, cols, rows])
            sample_length = u.shape[0]
            input_tensor = u[:sample_length//2].unsqueeze(0).unsqueeze(-2).to(device=device)
            target_tensor = u[sample_length//2:].unsqueeze(0).unsqueeze(-2).to(device=device)
        
        
    
        # Initialize and set up the model
        phycell  =  PhyCell(input_shape=(input_tensor.shape[-1]//4+1),
                            input_dim=input_dim,
                            F_hidden_dims=[7],
                            n_layers=1,
                            kernel_size=7,
                            device=device) 
        
        convcell =  ConvLSTM(input_shape=(input_tensor.shape[-1]//4+1),
                             input_dim=input_dim,
                             hidden_dims=hidden_dims,
                             n_layers=n_layers_convcell,
                             kernel_size=3,
                             device=device)
        
        model  = EncoderRNN(phycell,
                              convcell,
                              input_channels=1,
                              input_dim=(input_tensor.shape[-1],),
                              _1d=True,
                              bc=bc,
                              learn_BC=True,
                              device=device,
                              sigmoid=False,
                              small=config.model.small)
        
        constraints = th.zeros((7,7)).to(device)
        ind = 0
        for i in range(0,7):
            constraints[ind,i] = 1
            ind +=1
    
    if model.learn_BC:
        # Initialize lists to store
        left_BC = []
        right_BC = []
        
        left_BC_grad = []
        right_BC_grad = []
        
        mse_train = []
    
    if print_progress:
        print(phycell)
        print(convcell)
        print(model)
        
        # Count number of trainable parameters
        pytorch_total_params = sum(
            p.numel() for p in phycell.parameters() if p.requires_grad
        )
        print("PhyCell parameters:", pytorch_total_params)
        pytorch_total_params = sum(
            p.numel() for p in convcell.parameters() if p.requires_grad
        )
        print("ConvLSTM parameters:", pytorch_total_params)
        pytorch_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print("Total trainable model parameters:", pytorch_total_params)

    # If desired, restore the network by loading the weights saved in the .pt
    # file
    if config.training.continue_training:
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

    a = time.time()
    #
    # Start the training and iterate over all epochs
    for epoch in range(config.training.epochs):

        epoch_start_time = time.time()
        teacher_forcing_ratio = np.maximum(0 , 1 - epoch * 0.003)
        
        # Define the closure function that consists of resetting the
        # gradient buffer, loss function calculation, and backpropagation
        # It is necessary for LBFGS optimizer, because it requires multiple
        # function evaluations
        def closure():
            optimizer.zero_grad()
            # input_tensor : torch.Size([batch_size, input_length, channels, cols, rows])
            input_length  = input_tensor.size(1)
            target_length = target_tensor.size(1)
            mse = 0
            for ei in range(input_length-1): 
                encoder_output, encoder_hidden, output_image,_,_ = model(input_tensor[:,ei], ei==0)

                mse += criterion(output_image,input_tensor[:,ei+1])
        
            decoder_input = input_tensor[:,-1,:,:] # first decoder input = last image of input sequence
            
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False 
            for di in range(target_length):
                decoder_output, decoder_hidden, output_image,_,_ = model(decoder_input)
                target = target_tensor[:,di]
                mse += criterion(output_image,target)
                if use_teacher_forcing:
                    decoder_input = target # Teacher forcing
                else:
                    decoder_input = output_image

            mse.backward()
            
            return mse / target_tensor.size(1)
        
        optimizer.step(closure)
            
        # Extract the MSE value from the closure function
        mse = closure()
        
        if model.learn_BC:
            print(f"BC: {model.bc}")
            left_BC.append(float(model.bc[0,0,0]))
            right_BC.append(float(model.bc[0,0,1]))
            
            print(f"\n BC_grad: {model.bc.grad}")
            if np.abs(model.bc.grad[0,0,0]) > 5:
                left_BC_grad.append(float(5.0))
            else:
                left_BC_grad.append(float(model.bc.grad[0,0,0]))
                
            if np.abs(model.bc.grad[0,0,1]) > 5:
                right_BC_grad.append(float(5.0))
            else:
                right_BC_grad.append(float(model.bc.grad[0,0,1]))
            
            print(f"mse_train: {mse.item()}")
            mse_train.append(float(mse))
        
        epoch_errors_train.append(mse.item())

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
                    bc=np.array2string(model.bc.detach().numpy(), precision=5),
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
        ax[0].axhline(y=-1.0, color='saddlebrown')
        ax[0].axhline(y=1.0, color='saddlebrown')
        ax[0].grid(True)
        
        ax[1].plot(left_BC_grad, label='left BC gradient', color="red")
        ax[1].plot(right_BC_grad, label='right BC gradient', color="darkblue")
        ax[1].legend(loc="best", fontsize=18)
        ax[1].set_title("Convergence of the Gradients of the BC's", size=22)
        ax[1].grid(True)
        
        plt.tight_layout()
        plt.draw()
        plt.show()
        # plt.savefig(f"{config.model.name}.pdf")
        
        # Plot the convergence of the error during inference
        fig, ax = plt.subplots(1, figsize=(9, 7))
        
        ax.plot(mse_train, label='MSE During Training', color="red")
        ax.legend(loc="best", fontsize=18)
        ax.set_title("Convergence of the Error", size=22)
        ax.grid(True)
        
        plt.tight_layout()
        plt.draw()
        plt.show()
        # plt.savefig(f"{config.model.name}_error.pdf")
        plt.close()

if __name__ == "__main__":
    th.set_num_threads(1)
    run_training()

    print("Done.")