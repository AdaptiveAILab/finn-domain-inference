#! env/bin/python3

"""
Main file for testing (evaluating) a model
"""

import numpy as np
import torch as th
import torch.nn as nn
import glob
import os
import time
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

sys.path.append("..")
from utils.configuration import Configuration
import utils.helper_functions as helpers
from phydnet import ConvLSTM, PhyCell, EncoderRNN
from constrain_moments import K2M


def run_testing(print_progress=False, visualize=False, model_number=None, infer_BC=True):

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
        # Load samples, together with x, y, and t series
        t = np.load(os.path.join(data_path, "t_series.npy"))
        x = np.load(os.path.join(data_path, "x_series.npy"))
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                             dtype=th.float).to(device=device)
        
        u = u + th.normal(th.zeros_like(u),th.ones_like(u)*config.data.noise)
        
        bc = th.tensor([[[0.0, 0.0]]]).to(device)
        
        sample_length = u.shape[0]
        input_tensor = u[:sample_length//2].unsqueeze(0).unsqueeze(-2).to(device=device)
        target_tensor = u[sample_length//2:].unsqueeze(0).unsqueeze(-2).to(device=device)
        print(f"u: {u.shape}")
        print(f"input_tensor: {input_tensor.shape}")
        print(f"target_tensor: {target_tensor.shape}")
    
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
                              device=device,
                              sigmoid=False,
                              small=config.model.small)
        
        constraints = th.zeros((7,7)).to(device)
        ind = 0
        for i in range(0,7):
            constraints[ind,i] = 1
            ind +=1
                
    elif config.data.type == "allen_cahn":
        # Load samples, together with x, y, and t series
        t = np.load(os.path.join(data_path, "t_series.npy"))
        x = np.load(os.path.join(data_path, "x_series.npy"))
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                             dtype=th.float).to(device=device)
        
        u = u + th.normal(th.zeros_like(u),th.ones_like(u)*config.data.noise)
        
        bc = th.tensor([[[0.0, 0.0]]]).to(device)
        
        sample_length = u.shape[0]
        input_tensor = u[:sample_length//2].unsqueeze(0).unsqueeze(-2).to(device=device)
        target_tensor = u[sample_length//2:].unsqueeze(0).unsqueeze(-2).to(device=device)
        print(f"u: {u.shape}")
        print(f"input_tensor: {input_tensor.shape}")
        print(f"target_tensor: {target_tensor.shape}")
    
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
                              device=device,
                              sigmoid=False,
                              small=config.model.small)
        
        constraints = th.zeros((7,7)).to(device)
        ind = 0
        for i in range(0,7):
            constraints[ind,i] = 1
            ind +=1

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Trainable model parameters: {pytorch_total_params}\n")

    # Load the trained weights from the checkpoints into the model
    model_path = os.path.join(os.path.abspath(""),
                              "checkpoints",
                              config.model.name,
                              config.model.name+".pt")
    model.load_state_dict(th.load(model_path))
    
    # Block for inferring the BC-Values
    if infer_BC:
        
        # Set the model to train mode
        model.train()
        # Initialize the criterion (loss)
        criterion = nn.MSELoss(reduction="mean")
        
        # Set all trainable parameters to false
        for p in model.parameters():
            p.requires_grad = False
        
        if config.data.type == "burger":
            # Set BC as parameter
            model.bc = nn.Parameter(th.tensor([[[np.random.uniform(-1,1), np.random.uniform(-1,1)]]], dtype=th.float))
            #[[np.random.uniform(-1,1)], [np.random.uniform(-1,1)]]
            print(f"BC: {model.bc}")
        
        elif config.data.type == "allen_cahn":
            # Set BC as parameter
            model.bc = nn.Parameter(th.tensor([[[np.random.uniform(-0.3,0.3), np.random.uniform(-0.3,0.3)]]], dtype=th.float))
            print(f"BC: {model.bc}")
        
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
        iterations = 300
        
        # Slice data for the inference
        u_infer = u[:data_infer,:]
        sample_length_infer = u_infer.shape[0]
        input_tensor_infer = u_infer[:sample_length_infer//2].unsqueeze(0).unsqueeze(-2).to(device=device)
        target_tensor_infer = u_infer[sample_length_infer//2:].unsqueeze(0).unsqueeze(-2).to(device=device)
        print(f"u: {u.shape}")
        print(f"input_tensor: {input_tensor.shape}")
        print(f"target_tensor: {target_tensor.shape}")

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
                optimizer.zero_grad()
                
                # input_tensor : torch.Size([batch_size, input_length, channels, cols, rows]) 
                input_length  = input_tensor_infer.size(1)
                target_length = target_tensor_infer.size(1)
                
                mse = 0
                #print(f"input_tensor_infer: {input_tensor_infer.shape}")
                for ei in range(input_length-1): 
                    encoder_output, encoder_hidden, output_image,_,_ = model(input_tensor_infer[:,ei], (ei==0) )

                    mse += criterion(output_image,input_tensor_infer[:,ei+1])
                   
                decoder_input = input_tensor_infer[:,-1,:,:] # first decoder input = last image of input sequence
                
                for di in range(target_length):
                    decoder_output, decoder_hidden, output_image,_,_ = model(decoder_input)
                    target = target_tensor_infer[:,di]
                    
                    mse += criterion(output_image,target)   
                    
                    # The inference runs in closed-loop, no teacher forcing
                    decoder_input = output_image
            
                # Moment regularization  # encoder.phycell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
                #===============================================================
                # k2m = K2M([7]).to(device)
                # for b in range(0,model.phycell.cell_list[0].input_dim):
                #     filters = model.phycell.cell_list[0].F.conv1.weight[:,b,:] # (nb_filters,7,7)     
                #     m = k2m(filters.double()) 
                #     m  = m.float()   
                #     mse += criterion(m, constraints) # constrains is a precomputed matrix   
                #===============================================================
                mse.backward()
                
                return mse / target_tensor_infer.size(1)
        
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
            
            print(f"BC: {model.bc}")
            left_BC.append(float(model.bc[0,0,0]))
            right_BC.append(float(model.bc[0,0,1]))
            
            print(f"\n {model.bc.grad}")
            left_BC_grad.append(float(model.bc.grad[0,0,0]))
            right_BC_grad.append(float(model.bc.grad[0,0,1]))
            
        #Cut the used values from the data
        u = u[data_infer:,:]
        sample_length = u.shape[0]
        input_tensor = u[:sample_length//2].unsqueeze(0).unsqueeze(-2).to(device=device)
        target_tensor = u[sample_length//2:].unsqueeze(0).unsqueeze(-2).to(device=device)
        t = t[data_infer:]
        
        
    '''Evaluation'''
        
    model.eval()

    time_start = time.time()
    with th.no_grad():
    
        input_length = input_tensor.size()[1]
        print(f"input_length: {input_length}")
        target_length = target_tensor.size()[1]
        print(f"target_length: {target_length}")
    
        predictions = []
        for ei in range(input_length-1):
            encoder_output, encoder_hidden, output_image,_,_  = model(input_tensor[:,ei], (ei==0))
            predictions.append(output_image.cpu())
                
        decoder_input = input_tensor[:,-1,:,:] # first decoder input= last image of input sequence
    
        for di in range(target_length):
            decoder_output, decoder_hidden, output_image,_,_ = model(decoder_input, False, False)
            decoder_input = output_image
            predictions.append(output_image.cpu())
    
        input = input_tensor.cpu().numpy()
        target = target_tensor.cpu().numpy()
            
        target = np.concatenate((input,target),axis=1)
        target = target[:,1:]
            
        predictions =  np.stack(predictions) # (nt, batch_size, channels, Nx, Ny)
        predictions = predictions.swapaxes(0,1)  # (batch_size, nt, channels, Nx, Ny)
      
    if print_progress:
      print(f"Forward pass took: {time.time() - time_start} seconds.")

    mse = np.mean((predictions - target)**2)
    print(f"MSE_final: {mse}")
    print(f"BC_final: {model.bc}")
        
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
        
        u_hat = np.transpose(predictions.squeeze())
        u = np.transpose(target.squeeze())
     
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
        ax.set_ylim([-1.1, 1.1])
     
        anim = animation.FuncAnimation(fig,
                                       animate_1d,
                                       frames=len(t) - 1,
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
        
        u_hat = np.transpose(predictions.squeeze())
        u = np.transpose(target.squeeze())
     
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
        ax.set_ylim([-1.1, 1.1])
     
        anim = animation.FuncAnimation(fig,
                                       animate_1d,
                                       frames=len(t) - 1,
                                       fargs=(line1, line2, u, u_hat),
                                       interval=20)
        plt.tight_layout()
        plt.draw()
        plt.show()

    return mse, model.bc


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
    im1.set_array(u_hat[t,:,:].squeeze().transpose())
    im2.set_array(u[t,:,:].squeeze().transpose())


if __name__ == "__main__":
    th.set_num_threads(1)
    
    run_testing(print_progress=True, visualize=True)

    print("Done.")