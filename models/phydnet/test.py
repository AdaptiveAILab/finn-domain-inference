#! env/bin/python3

"""
Main file for testing (evaluating) a model
"""

import numpy as np
import torch as th
import torch.nn as nn
import glob
import os
from threading import Thread
import time
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

sys.path.append("..")
from utils.configuration import Configuration
import utils.helper_functions as helpers
from phydnet import ConvLSTM, PhyCell, EncoderRNN
from constrain_moments import K2M


def run_testing(print_progress=True, visualize=True, model_number=None, infer_BC=False, infer_domain=False):
    # Note that it is not possible to switch on only infer_domain without infer_BC
    if infer_domain and not infer_BC:
        raise AssertionError("There is no scenario with known BC and unknown domain.")

    # Load the user configurations
    config = Configuration("config.json")
    
    # Append the model number to the name of the model
    if model_number is None:
        model_number = config.model.number
    config.model.name = config.model.name + "_" + str(model_number).zfill(2)

    # Print some information to console
    print("Model name:", config.model.name)

    # Hide the GPU(s) in case the user specified to use the CPU in the config file
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
                              learn_BC=False,
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

    # Initialize the criterion (loss)
    criterion = nn.MSELoss()

    '''  INFERENCE  '''

    if infer_BC:
        # Set all trainable parameters to false
        for p in model.parameters():
            p.requires_grad = False
        
        if config.data.type == "burger":
            # Set BC as parameter
            model.bc = nn.Parameter(th.tensor([[[np.random.uniform(-1,1), np.random.uniform(-1,1)]]], dtype=th.float))
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

        if infer_domain:
            cut_domain_size = 10
            R = 10

            # Create retrospective reconstruction domain
            model.retro_domain = nn.Parameter(th.zeros((1, 1, u.shape[-1]), dtype=th.float)).to(device=device)

            # Count number of trainable parameters
            pytorch_total_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            print(f"Trainable model parameters with the unseen data: {pytorch_total_params}\n")

        # If desired, restore the network weights after inference
        if config.inference.continue_inference:
            print("Restoring the inference model.")
            model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                                       "checkpoints",
                                                       'infer_' + config.model.name,
                                                       'infer_' + config.model.name + ".pt")))

        # The number of time steps for inference
        data_infer = 80
        
        # Set the number of iterations
        iterations = 1

        # Add noise to the data
        u[:data_infer] = u[:data_infer] + th.normal(th.zeros_like(u[:data_infer]),
                                                    th.ones_like(u[:data_infer]) * config.inference.noise)

        # Slice data for the inference
        u_infer = u[:data_infer,:]
        sample_length_infer = u_infer.shape[0]
        input_tensor_infer = u_infer[:sample_length_infer//2].unsqueeze(0).unsqueeze(-2).to(device=device)
        target_tensor_infer = u_infer[sample_length_infer//2:].unsqueeze(0).unsqueeze(-2).to(device=device)
        print(f"u_infer: {u_infer.shape}")
        print(f"input_tensor_infer: {input_tensor_infer.shape}")
        print(f"target_tensor_infer: {target_tensor_infer.shape}")

        # Initialize lists to store
        epoch_errors_infer = []
        best_infer = np.infty

        left_BC = []
        right_BC = []
        
        left_BC_grad = []
        right_BC_grad = []

        # Set up an optimizer
        optimizer = th.optim.Adam(model.parameters(), lr=config.training.learning_rate)
        
        for i in range(iterations):
            # Define the closure function that consists of resetting the
            # gradient buffer, loss function calculation, and backpropagation
            # It is necessary for LBFGS optimizer, because it requires multiple
            # function evaluations
            # It can be used for Adam as well.
            def closure():
                # Set model to train mode
                model.train()

                # Reset the optimizer to clear data from previous iterations
                optimizer.zero_grad()
                
                # input_tensor : torch.Size([batch_size, input_length, channels, cols, rows])
                input_length  = input_tensor_infer.size(1)
                target_length = target_tensor_infer.size(1)

                # Initialize loss
                loss = 0.0
                if infer_domain:
                    for ei in range(input_length - 1 + R):
                        if ei == 0:
                            # Feed the retrospective initial state into the model
                            encoder_output, encoder_hidden, output_image, _, _ = model(model.retro_domain, ei == 0)
                        else:
                            # Feed the inferred domain into the model
                            # No teacher forcing in this setting, only close-loop forward propagation
                            encoder_output, encoder_hidden, output_image, _, _ = model(output_image, ei == 0)

                        if ei < R:
                            # No loss for the past
                            pass
                        else:
                            # Compute the loss only for the inner "seen" domain
                            loss = loss + criterion(output_image[0,0,cut_domain_size:-cut_domain_size],
                                                          input_tensor_infer[0, ei -R + 1, 0, cut_domain_size:-cut_domain_size])

                    # First decoder input is the last image of input sequence combined with the inferred domain
                    decoder_input = th.cat(
                                (output_image[:,:,:cut_domain_size],
                                 input_tensor_infer[:, -1, :, cut_domain_size:-cut_domain_size],
                                 output_image[:, :, -cut_domain_size:]), dim=-1)

                    for di in range(target_length):
                        decoder_output, decoder_hidden, output_image, _, _ = model(decoder_input)
                        target = target_tensor_infer[:, di]

                        # Compute loss only for the inner "seen" domain
                        loss = loss + criterion(output_image[0,0,cut_domain_size:-cut_domain_size],
                                          target[0,0,cut_domain_size:-cut_domain_size])

                        # The inference runs in close-loop, no teacher forcing
                        decoder_input = output_image
                else:
                    for ei in range(input_length-1):

                        encoder_output, encoder_hidden, output_image,_,_ = model(input_tensor_infer[:, ei], ei==0)

                        loss = loss + criterion(output_image,input_tensor_infer[:,ei+1])

                    # First decoder input is the last image of the input sequence
                    decoder_input = input_tensor_infer[:,-1,:,:]

                    for di in range(target_length):
                        decoder_output, decoder_hidden, output_image,_,_ = model(decoder_input)
                        target = target_tensor_infer[:,di]

                        loss = loss + criterion(output_image,target)

                        # The inference runs in closed-loop, no teacher forcing
                        decoder_input = output_image

                loss.backward()
                
                return loss / target_tensor_infer.size(1)
        
            optimizer.step(closure)
                
            # Extract the MSE value from the closure function
            loss = closure()
            
            epoch_errors_infer.append(loss.item())
            
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
                        bc=np.array2string(model.bc.detach().numpy(), precision=5),
                        net=model,
                        infer=True))
                    thread.start()

            print(f"Loss: {infer_sign}{loss} \n Iteration: {i}")
            
            print(f"BC: {model.bc}")
            left_BC.append(float(model.bc[0,0,0]))
            right_BC.append(float(model.bc[0,0,1]))
            
            print(f"\n {model.bc.grad}")
            left_BC_grad.append(float(model.bc.grad[0,0,0]))
            right_BC_grad.append(float(model.bc.grad[0,0,1]))
        #### Inference loop is over ####

        if not infer_domain:
            #Cut the used values from the data
            u_eval = u[data_infer:,:]
            sample_length = u_eval.shape[0]
            input_tensor = u_eval[:sample_length//2].unsqueeze(0).unsqueeze(-2).to(device=device)
            target_tensor = u_eval[sample_length//2:].unsqueeze(0).unsqueeze(-2).to(device=device)
        
    '''Evaluation'''
    if infer_BC or infer_domain:
        print('Restoring the model')
        model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                                   "checkpoints",
                                                   'infer_' + config.model.name,
                                                   'infer_' + config.model.name + ".pt")))
    model.eval()

    # Print the inferred BC
    if infer_BC:
        print(f"BC_final: {model.bc}")

    time_start = time.time()

    # Forward data through the model
    with th.no_grad():
        input_length = input_tensor.size()[1]
        print(f"input_length: {input_length}")
        target_length = target_tensor.size()[1]
        print(f"target_length: {target_length}")
    
        predictions = []
        if infer_domain:
            for ei in range(input_length - 1 + R):
                if ei == 0:
                    predictions.append(model.retro_domain.cpu())

                    # Feed the retrospective initial state into the model
                    encoder_output, encoder_hidden, output_image, _, _ = model(model.retro_domain, ei == 0)
                    predictions.append(output_image.cpu())

                else:
                    # Feed the inferred domain into the model
                    # No teacher forcing in this setting, only close-loop forward propagation
                    encoder_output, encoder_hidden, output_image, _, _ = model(output_image, ei == 0)
                    predictions.append(output_image.cpu())

            # First decoder input is the last image of input sequence combined with the inferred domain
            decoder_input = th.cat(
                (output_image[:, :, :cut_domain_size],
                 input_tensor[:, -1, :, cut_domain_size:-cut_domain_size],
                 output_image[:, :, -cut_domain_size:]), dim=-1)

            for di in range(target_length):
                decoder_output, decoder_hidden, output_image, _, _ = model(decoder_input, False, False)
                decoder_input = output_image
                predictions.append(output_image.cpu())
        else:
            for ei in range(input_length-1):
                encoder_output, encoder_hidden, output_image,_,_  = model(input_tensor[:,ei], (ei==0))
                predictions.append(output_image.cpu())

            # First decoder input is the last image of input sequence
            decoder_input = input_tensor[:,-1,:,:]

            for di in range(target_length):
                decoder_output, decoder_hidden, output_image,_,_ = model(decoder_input, False, False)
                decoder_input = output_image
                predictions.append(output_image.cpu())
    
        input = input_tensor.cpu().numpy()
        target = target_tensor.cpu().numpy()
            
        target = np.concatenate((input,target),axis=1)
        if not infer_domain:
            target = target[:,1:]

        predictions =  np.stack(predictions) # (nt, batch_size, channels, Nx, Ny)
        predictions = predictions.swapaxes(0,1)  # (batch_size, nt, channels, Nx, Ny)
      
    if print_progress:
      print(f"Forward pass took: {time.time() - time_start} seconds.")

    if infer_domain:
        # Save certain inferred retrospective domains
        # In case R=10
        # inferred_retro_domains = {
        #     # model.retro_domain is the same as u_hat[0]
        #     f'({config.model.name}: t={-R})': [model.retro_domain.detach().numpy()],
        #     f'(t={-R + 1})': [predictions[0,1,0,:]],
        #     f'(t={-R + 2})': [predictions[0,2,0,:]],
        #     f'(t={-R + 3})': [predictions[0,3,0,:]],
        #     f'(t={-R + 4})': [predictions[0,4,0,:]],
        #     f'(t={-R + 5})': [predictions[0,5,0,:]],
        #     f'(t={-R + 6})': [predictions[0,6,0,:]],
        #     f'(t={-R + 7})': [predictions[0,7,0,:]],
        #     f'(t={-R + 8})': [predictions[0,8,0,:]],
        #     f'(t={-R + 9})': [predictions[0,9,0,:]],
        #     f'(t={-R + R})': [predictions[0,R,0,:]]
        # }

        # In case R=5
        # inferred_retro_domains = {
        #     # model.retro_domain is the same as u_hat[0]
        #     f'({config.model.name}: t={-R})': [model.retro_domain.detach().numpy()],
        #     f'(t={-R + 1})': [predictions[0,1,0,:]],
        #     f'(t={-R + 3})': [predictions[0,3,0,:]],
        #     f'(t={-R + 4})': [predictions[0,4,0,:]],
        #     f'(t={-R + R})': [predictions[0,R,0,:]]
        # }

        # Print the inference at t = 0
        inferred_retro_domains = {
            f'({config.model.name}: t={-R + R})': [predictions[0,R,0,:]]
        }
        print(f"inferred_retro_domains: {inferred_retro_domains}")

        # Discard the past
        predictions = predictions[:,R:]

    # Compute test prediction error
    if infer_domain:
        # Exclude the inference data from the prediction of the error computation
        # Whole-Domain Prediction Error
        # mse = np.mean((predictions[:,data_infer:] - target[:,data_infer:])**2)

        # Seen-Domain Prediction Error
        mse = np.mean((predictions[:,data_infer:,:,cut_domain_size:-cut_domain_size] -
                       target[:,data_infer:,:,cut_domain_size:-cut_domain_size]) ** 2)
    else:
        mse = np.mean((predictions - target)**2)
    print(f"MSE_final: {mse}")

    #
    # Visualize the data
    if visualize:
        if config.data.type == "burger" or config.data.type == "allen_cahn":
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

                plt.tight_layout()
                plt.draw()
                plt.show()
                # plt.savefig(f"{config.model.name}.pdf")

                # Plot the convergence of the error during inference
                fig, ax = plt.subplots(1, figsize=(13, 7))

                ax.plot(np.log(epoch_errors_infer), label='MSE During Inference', color="red")
                ax.legend(loc="best", fontsize=18)
                ax.set_title("Convergence of the Error", size=26)
                ax.set_xlabel('Iterations', size=22)
                ax.set_ylabel('Log-Error', size=22)

                ax.grid(True, linewidth=0.5)

                plt.tight_layout()
                plt.draw()
                plt.show()
                # plt.savefig(f"{config.model.name}_error.pdf")

            u_hat = predictions.squeeze()
            u = u.detach().numpy()

            # u(t, x) over space
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            plt.style.use('ggplot')

            if not infer_domain and infer_BC:
                ax[0].imshow(np.flip(np.concatenate((u[:data_infer + 1].copy(),
                                                     u_hat.copy())), 0),
                             interpolation='nearest',
                             extent=[x.min(), x.max(),
                                     t.min(), t.max()],
                             origin='upper', aspect='auto')
                ax[0].axhline(y=t[data_infer], color='red')
            else:
                ax[0].imshow(np.flip(u_hat.copy(), 0),
                             interpolation='nearest',
                             extent=[x.min(), x.max(),
                                     t.min(), t.max()],
                             origin='upper', aspect='auto')
                if config.inference.noise > 0.0 and infer_domain:
                    ax[0].axhline(y=t[data_infer], color='red')

            ax[0].set_ylim(t.min(), t.max())
            ax[0].set_xlim(x.min(), x.max())
            ax[0].set_xticks([x.min(), 0, x.max()])
            ax[0].set_xticklabels([f'{x.min()}', '0', f'{x.max()}'])
            ax[0].set_yticks([t.min(), 0.5, t.max()])
            ax[0].set_yticklabels([f'{t.min()}', '0.5', f'{t.max()}'])
            ax[0].set_xlabel('$x$', size=22)
            ax[0].set_ylabel('$t$', size=22)
            ax[0].set_title('PhyDNet', fontsize=26)
            ax[0].grid(False)

            h = ax[1].imshow(np.flip(u.copy(), 0),
                             interpolation='nearest',
                             extent=[x.min(), x.max(),
                                     t.min(), t.max()],
                             origin='upper', aspect='auto')
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax, ticks=[-0.8, 0, 0.8])
            cbar.ax.set_yticklabels(['-0.8', '0', '0.8'])

            if config.inference.noise > 0.0 and infer_domain:
                ax[1].axhline(y=t[data_infer], color='red')
            ax[1].set_ylim(t.min(), t.max())
            ax[1].set_xlim(x.min(), x.max())
            ax[1].set_xticks([x.min(), 0, x.max()])
            ax[1].set_xticklabels([f'{x.min()}', '0', f'{x.max()}'])
            ax[1].set_yticks([t.min(), 0.5, t.max()])
            ax[1].set_yticklabels([f'{t.min()}', '0.5', f'{t.max()}'])
            ax[1].set_xlabel('$x$', size=22)
            ax[1].set_ylabel('$t$', size=22)
            ax[1].set_title('DATA', fontsize=26)
            ax[1].grid(False)

            plt.tight_layout()
            plt.draw()
            plt.show()
            # plt.savefig(f"{config.model.name}_space.pdf")

            # u(t, x) over time
            u_hat = np.transpose(u_hat)
            u = np.transpose(u)

            if not infer_domain and infer_BC:
                u = u[:, data_infer + 1:]
                t = t[data_infer:]

            fig, ax = plt.subplots()
            line1, = ax.plot(x, u[:, 0], 'ro', linewidth=2, label='Data')
            line2, = ax.plot(x, u_hat[:, 0], 'b-', linewidth=2, label='Prediction')
            ax.set_xlabel('$x$', fontsize=24)
            ax.set_ylabel('$u(t,x)$', fontsize=24)
            ax.set_xlim([x.min(), x.max()])
            # Choose lim according to the data BC
            ax.set_ylim([-3.4, 3.4])
            ax.legend(loc="upper left", fontsize=16)
            ax.grid(True, linewidth=0.3)
            if infer_domain:
                ax.axvline(x=x[cut_domain_size], color='saddlebrown', linewidth=2.5)
                ax.axvline(x=x[-cut_domain_size-1], color='saddlebrown', linewidth=2.5)

            anim = animation.FuncAnimation(fig,
                                           animate_1d,
                                           frames=len(t) - 1,
                                           fargs=(line1, line2, u, u_hat),
                                           interval=20)

            plt.tight_layout()
            plt.draw()
            plt.show()

            # f = f"{config.model.name}_animation.mp4"
            # writervideo = animation.FFMpegWriter(fps=60)
            # anim.save(f, writer=writervideo)

            # u(t,x) at one specific t
            fig, ax = plt.subplots(sharex=True, figsize=(6,4))
            ax.plot(x, u[:, -1], 'ro', linewidth=2.0)  # , label='Exact')
            ax.plot(x, u_hat[:, -1], 'b-', linewidth=2.0)  # , label='Prediction')

            ax.set_xlabel('$x$', fontsize=24)
            ax.set_ylabel(f'$u(x, t={int(t.max())})$', fontsize=24)
            # ax.set_xlim([x.min(), x.max()])
            # ax.set_ylim([-1.4, 1.4])
            ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            # ax.set_xticks([x.min(), 0, x.max()])

            ax.grid(True, linewidth=0.3)
            if infer_domain:
                ax.axvline(x=x[cut_domain_size], color='saddlebrown', linewidth=2.5)
                ax.axvline(x=x[-cut_domain_size - 1], color='saddlebrown', linewidth=2.5)
            plt.tight_layout()
            plt.draw()
            plt.show()
            # plt.savefig(f"{config.model.name}_time.pdf")
            plt.close('all')

    if infer_domain:
        return mse, model.bc, inferred_retro_domains
    else:
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


if __name__ == "__main__":
    th.set_num_threads(1)
    # Uncomment the next line if infer_domain=True
    # mse, inferred_BC, inferred_retro_domains = run_testing()

    # Uncomment the next line if infer_domain=False
    mse, inferred_BC = run_testing()

    print("Done.")