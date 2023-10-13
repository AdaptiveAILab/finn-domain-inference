import numpy as np
import torch as th
import torch.nn as nn
import time
import glob
import os
from threading import Thread
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

sys.path.append("..")
from utils.configuration import Configuration
import utils.helper_functions as helpers
from distana import DISTANA


def run_testing(print_progress=True, visualize=True, model_number=None, infer_BC=False, infer_domain=False):
    # Note that it is not possible to switch on only infer_domain without infer_BC
    if infer_domain and infer_BC==False:
        raise AssertionError("There is no scenario with known BC and unknown domain.")
    
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

    # Set device on GPU if specified in the configuration file, else CPU
    device = config.general.device #helpers.determine_device(print_progress=False)
    
    #
    # Load the data depending on the task
    if config.data.type == "burger":
        root_path = os.path.abspath("../../data")
        data_path = os.path.join(root_path, config.data.type, config.data.name)
        
        # Load samples, together with x, y, and t series
        t_series = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
                      
        x = np.load(os.path.join(data_path, "x_series.npy"))
        
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                             dtype=th.float).to(device=device)
        
        data = u.unsqueeze(1).unsqueeze(1)
        
        bc = th.tensor([[[0.0, 0.0]]], device=device)

        
    elif config.data.type == "allen_cahn":
        root_path = os.path.abspath("../../data")
        data_path = os.path.join(root_path, config.data.type, config.data.name)
        
        # Load samples, together with x, y, and t series
        t_series = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
                      
        x = np.load(os.path.join(data_path, "x_series.npy"))
        
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                             dtype=th.float).to(device=device)
        
        data = u.unsqueeze(1).unsqueeze(1)
        
        bc = th.tensor([[[0.0, 0.0]]], device=device)
        

    # Initialize and set up the network
    model = DISTANA(config=config, device=device, bc=bc, learn_BC=False).to(device=device)

    if print_progress:
        # Count number of trainable parameters
        pytorch_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print("Trainable model parameters:", pytorch_total_params)

        # Restore the network by loading the weights saved in the .pt file
        print("Restoring model (that is the network\'s weights) from file...")

    model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                  "checkpoints",
                                  config.model.name,
                                  config.model.name + ".pt")))

    # Initialize the criterion (loss)
    criterion = nn.MSELoss()

    '''  INFERENCE  '''

    if infer_BC:
        # Define method that forward propagates the data in the inference loop
        # The following method is used in case infer_domain=True
        def process_sequence_infer_domain(model, data, R, retro_domain,
                                          batch_size, config, device):
            
            # Set up an array of zeros to store the network outputs
            net_outputs = th.zeros(size=(len(data)+R,
                                         config.training.batch_size,
                                         config.model.dynamic_channels[-1],
                                         *config.model.field_size),
                                   device=device)
            
            model.reset(batch_size=batch_size)
            
            net_outputs[0] = retro_domain
            state_list = None
            
            # Iterate over the whole sequence of the training example and
            # perform a forward pass
            for t in range(1, len(data)+R):
                
                # Closed loop. There is no teacher forcing at inference
                net_input = net_outputs[t - 1]
                
                # Forward the input through the network
                #net_output, _ = model.forward(input_tensor=net_inputs[t])
                net_output, state_list = model.forward(input_tensor=net_input,
                                                       cur_state_list=state_list)
                
                # Store the output of the network for this sequence step
                net_outputs[t] = net_output
            
            return net_outputs
            
            
        # The following method is used in case infer_domain=False
        def process_sequence(model, criterion, data, batch_size, config,
                     device):
            
            # Separate the data into network inputs and labels
            net_inputs = data[:-1]
            net_labels = data[1:]
        
            # Set up an array of zeros to store the network outputs
            net_outputs = th.zeros(size=(len(net_labels),
                                         config.training.batch_size,
                                         config.model.dynamic_channels[-1],
                                         *config.model.field_size),
                                   device=device)
            
            model.reset(batch_size=batch_size)
        
            # Initial network input and forward pass
            net_output, state_list = model.forward(input_tensor=net_inputs[0],
                                                   cur_state_list=None)
            
            net_outputs[0] = net_output
        
            # Iterate over the whole sequence of the training example and
            # perform a forward pass
            for t in range(1, len(net_inputs)):
                
                # Closed loop. There is no teacher forcing at inference
                net_input = net_outputs[t - 1]
        
                # Forward the input through the network
                #net_output, _ = model.forward(input_tensor=net_inputs[t])
                net_output, state_list = model.forward(input_tensor=net_input,
                                                       cur_state_list=state_list)
                
                # Store the output of the network for this sequence step
                #net_outputs[t] = net_output[-1]
                net_outputs[t] = net_output
        
            # Compute the mean squared error
            mse = criterion(net_outputs, net_labels)
    
            return net_outputs, mse


        # Set a random seed for varying weight initializations
        th.seed()
        
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

        # Sets the masked data as learnable parameters in case wanted
        if infer_domain:
            cut_domain_size = 10
            R = 10

            if R == 0:
                # Set the parts of the initial state as learnable parameters for domain reconstruction
                model.init_left = nn.Parameter(th.zeros((1, 1, cut_domain_size), dtype=th.float)).to(device=device)
                model.init_right = nn.Parameter(th.zeros((1, 1, cut_domain_size), dtype=th.float)).to(device=device)
            else:
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
        
        # Set the number of iteration
        iterations = 1

        # Add noise to the data
        data[:data_infer] = data[:data_infer] + th.normal(th.zeros_like(data[:data_infer]),
                                                    th.ones_like(data[:data_infer]) * config.inference.noise)

        # Slice data for the inference
        u_infer = data[:data_infer,:]

        # Initialize lists to store
        epoch_errors_infer = []
        best_infer = np.infty

        left_BC = []
        right_BC = []
        
        left_BC_grad = []
        right_BC_grad = []
        
        # Set up an optimizer
        optimizer = th.optim.Adam(model.parameters(), lr=config.training.learning_rate, eps=1e-4)
        
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
                
                if infer_domain:
                    if R == 0:
                        # Concatanate 'seen' initial state with learnable parameters
                        model.retro_domain = th.cat(
                        (model.init_left,
                         u_infer[0,:,:,cut_domain_size:-cut_domain_size],
                         model.init_right), dim=-1)

                    # Forward propagation
                    net_outputs = process_sequence_infer_domain(
                        model=model,
                        data=u_infer,
                        R = R,
                        retro_domain=model.retro_domain,
                        batch_size=config.training.batch_size,
                        config=config,
                        device=device
                    )

                    # Compute loss only for the inner "seen" domain omitting the retrospective inference
                    loss = criterion(net_outputs[R:,0,0,cut_domain_size:-cut_domain_size],
                                           u_infer[:,0,0,cut_domain_size:-cut_domain_size])
                else:
                    # Forward propagation and loss computation
                    _, loss = process_sequence(
                        model=model,
                        criterion=criterion,
                        data=u_infer,
                        batch_size=config.training.batch_size,
                        config=config,
                        device=device
                    )
                
                loss.backward()
                    
                return loss
            
            optimizer.step(closure)
                
            # Extract the MSE value from the closure function
            loss = closure()

            # Append the error to the error list
            epoch_errors_infer.append(loss.item())

            # Create a plus or minus sign for the inference error
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

    
    '''Evaluation'''
    # Restore the model
    if infer_BC or infer_domain:
        print('Restoring the model')
        model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                                   "checkpoints",
                                                   'infer_' + config.model.name,
                                                   'infer_' + config.model.name + ".pt")))

    # Concatanate data and the inferred domain one last time
    if infer_domain:
        if R == 0:
            model.retro_domain = th.cat(
                (model.init_left,
                 u_infer[0, :, :, cut_domain_size:-cut_domain_size],
                 model.init_right), dim=-1)

    model.eval()
    
    # Print the inferred BC
    if infer_BC:
        print(f"BC_final {model.bc}")

    # Forward data through the model
    if infer_domain:
        time_start = time.time()
        net_outputs = process_sequence_infer_domain(model=model,
                                                    data=th.clone(data),
                                                    R=R,
                                                    retro_domain=model.retro_domain,
                                                    batch_size=config.training.batch_size,
                                                    config=config,
                                                    device=device
                                                    )

        # Save certain inferred retrospective domains
        # In case R=10
        # inferred_retro_domains = {
        #     # model.retro_domain is the same as net_outputs[0]
        #     f'({config.model.name}: t={-R})': [model.retro_domain[0,0,:].detach().numpy()],
        #     f'(t={-R + 1})': [net_outputs[1, 0, 0, :]],
        #     f'(t={-R + 2})': [net_outputs[2, 0, 0, :]],
        #     f'(t={-R + 3})': [net_outputs[3, 0, 0, :]],
        #     f'(t={-R + 4})': [net_outputs[4, 0, 0, :]],
        #     f'(t={-R + 5})': [net_outputs[5, 0, 0, :]],
        #     f'(t={-R + 6})': [net_outputs[6, 0, 0, :]],
        #     f'(t={-R + 7})': [net_outputs[7, 0, 0, :]],
        #     f'(t={-R + 8})': [net_outputs[8, 0, 0, :]],
        #     f'(t={-R + 9})': [net_outputs[9, 0, 0, :]],
        #     f'(t={-R + R})': [net_outputs[R, 0, 0, :]]
        # }

        # In case R=5
        # inferred_retro_domains = {
        #     # model.retro_domain is the same as u_hat[0]
        #     f'({config.model.name}: t={-R})': [model.retro_domain[0,0,:].detach().numpy()],
        #     f'(t={-R + 1})': [net_outputs[1, 0, 0, :]],
        #     f'(t={-R + 3})': [net_outputs[3, 0, 0, :]],
        #     f'(t={-R + 4})': [net_outputs[4, 0, 0, :]],
        #     f'(t={-R + R})': [net_outputs[R, 0, 0, :]]
        # }

        # In case R=0
        # inferred_retro_domains = {
        #     f'{config.model.name}: t = {R}': [model.retro_domain[0,0,:].detach().numpy()]
        # }

        # Print the inference at t = 0
        inferred_retro_domains = {
            f'{config.model.name}: t = {-R + R}': [net_outputs[R, 0, 0, :]]
        }
        print(f"inferred_retro_domains: {inferred_retro_domains}")

        #Discard the past
        net_outputs = net_outputs[R:]
    else:
        if infer_BC:
            data_test = th.clone(data[data_infer:, :])
        else:
            data_test = th.clone(data)
        sequence_length = len(data_test) - 1
    
        # Evaluate the network for the given test data
    
        # Separate the data into network inputs and labels
        # These are not used if infer_domain=True
        net_inputs = th.clone(data_test[:-1])
        net_labels = th.clone(data_test[1:])
        
        # Set up an array of zeros to store the network outputs
        net_outputs = th.zeros(size=(sequence_length,
                                     config.testing.batch_size,                                 
                                     config.model.dynamic_channels[-1],
                                     *config.model.field_size),
                               device=device)
        state_list = None
    
        # Iterate over the remaining sequence of the test example
        # and perform a forward pass
        time_start = time.time()
        for t in range(len(net_inputs)):
    
            if t == 0:
                # Initial input
                net_input = net_inputs[t]
            else:
                # Closed loop
                net_input = net_outputs[t - 1]
                
            #=======================================================================
            # if t < config.testing.teacher_forcing_steps:
            #     # Teacher forcing
            #     net_input = net_inputs[t]
            # else:
            #     # Closed loop
            #     net_input = net_outputs[t - 1]
            #=======================================================================
    
            net_output, state_list = model.forward(input_tensor=net_input,
                                                   cur_state_list=state_list)

            net_outputs[t] = net_output

    if print_progress:
        forward_pass_duration = time.time() - time_start
        print("Forward pass took:", forward_pass_duration, "seconds.")

    # Convert the PyTorch network output tensor into a numpy array
    net_outputs = net_outputs.cpu().detach().numpy()[:, 0, 0]
    if infer_domain:
        net_labels = data.cpu().detach().numpy()[:, 0, 0]
    else:
        net_labels = net_labels.cpu().detach().numpy()[:, 0, 0]
    
    #
    # Compute test error
    if infer_domain:
        # Exclude the inference data from the prediction of the error computation
        # Whole-Domain Prediction Error
        # mse = np.mean(np.square(net_outputs[data_infer:,:] - net_labels[data_infer:,:]))

        # Seen-Domain Prediction Error
        mse = np.mean(np.square(net_outputs[data_infer:, cut_domain_size:-cut_domain_size] -
                                net_labels[data_infer:, cut_domain_size:-cut_domain_size]))
    else:
        mse = np.mean(np.square(net_outputs - net_labels))
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

            u_hat = net_outputs
            u = data[:, 0, 0, :].detach().numpy()

            # u(t, x) over space
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))

            if not infer_domain and infer_BC:
                ax[0].imshow(np.flip(np.concatenate((u[:data_infer+1].copy(),
                                                     u_hat.copy())), 0),
                             interpolation='nearest',
                             extent=[x.min(), x.max(),
                                     t_series.min(), t_series.max()],
                             origin='upper', aspect='auto')
                ax[0].axhline(y=t_series[data_infer], color='red')
            else:
                ax[0].imshow(np.flip(u_hat.copy(), 0),
                             interpolation='nearest',
                             extent=[x.min(), x.max(),
                                     t_series.min(), t_series.max()],
                             origin='upper', aspect='auto')
                if config.inference.noise > 0.0 and infer_domain:
                    ax[0].axhline(y=t_series[data_infer], color='red')

            ax[0].set_ylim(t_series.min(), t_series.max())
            ax[0].set_xlim(x.min(), x.max())
            ax[0].set_xticks([x.min(), 0, x.max()])
            ax[0].set_xticklabels([f'{x.min()}', '0', f'{x.max()}'])
            ax[0].set_yticks([t_series.min(), 0.5, t_series.max()])
            ax[0].set_yticklabels([f'{t_series.min()}', '0.5', f'{t_series.max()}'])
            ax[0].set_xlabel('$x$', size=22)
            ax[0].set_ylabel('$t$', size=22)
            ax[0].set_title('DISTANA', fontsize=26)
            ax[0].grid(False)

            h = ax[1].imshow(np.flip(u.copy(), 0),
                             interpolation='nearest',
                             extent=[x.min(), x.max(),
                                     t_series.min(), t_series.max()],
                             origin='upper', aspect='auto')
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax, ticks=[-0.8, 0, 0.8])
            cbar.ax.set_yticklabels(['-0.8', '0', '0.8'])

            if config.inference.noise > 0.0 and infer_domain:
                ax[1].axhline(y=t_series[data_infer], color='red')
            ax[1].set_ylim(t_series.min(), t_series.max())
            ax[1].set_xlim(x.min(), x.max())
            ax[1].set_xticks([x.min(), 0, x.max()])
            ax[1].set_xticklabels([f'{x.min()}', '0', f'{x.max()}'])
            ax[1].set_yticks([t_series.min(), 0.5, t_series.max()])
            ax[1].set_yticklabels([f'{t_series.min()}', '0.5', f'{t_series.max()}'])
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
                u = u[:, data_infer+1:]
                t_series = t_series[data_infer:]

            fig, ax = plt.subplots()
            line1, = ax.plot(x, u[:, 0], 'ro', linewidth=2, label='Data')
            line2, = ax.plot(x, u_hat[:, 0], 'b-', linewidth=2, label='Prediction')
            ax.set_xlabel('$x$', fontsize=24)
            ax.set_ylabel('$u(t,x)$', fontsize=24)
            ax.set_xlim([x.min(), x.max()])
            ax.set_ylim([-1.4, 1.4])
            ax.legend(loc="upper left", fontsize=16)
            ax.grid(True, linewidth=0.3)
            if infer_domain:
                ax.axvline(x=x[cut_domain_size], color='saddlebrown', linewidth=2.5)
                ax.axvline(x=x[-cut_domain_size-1], color='saddlebrown', linewidth=2.5)

            anim = animation.FuncAnimation(fig,
                                           animate_1d,
                                           frames=len(t_series) - 1,
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
            ax.set_ylabel(f'$u(x, t={int(t_series.max())})$', fontsize=24)
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


def animate(t, axis, field):
    """
    Data animation function animating an image over time.
    :param t: The current time step
    :param axis: The matplotlib image object
    :param field: The data field
    :return: The matplotlib image object updated with the current time step's
        image date
    """

    axis.set_ydata(field[:, t])


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
    im1.set_array(u_hat[t, :, :].squeeze().t().detach())
    im2.set_array(u[t, :, :].squeeze().t().detach())


if __name__ == "__main__":
    th.set_num_threads(1)
    # Uncomment the next line if infer_domain=True
    # mse, inferred_BC, inferred_retro_domains = run_testing()

    # Uncomment the next line if infer_domain=False
    mse, inferred_BC = run_testing()

    print("Done.")