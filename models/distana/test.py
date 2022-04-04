import numpy as np
import torch as th
import torch.nn as nn
import time
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

sys.path.append("..")
from utils.configuration import Configuration
import utils.helper_functions as helpers
from distana import DISTANA


def run_testing(print_progress=False, visualize=False, model_number=None, infer_BC=True):

    th.set_num_threads(1)
    
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
    model = DISTANA(config=config, device=device, bc=bc).to(device=device)

    if print_progress:
        # Count number of trainable parameters
        pytorch_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print("Trainable model parameters:", pytorch_total_params)

        # Restore the network by loading the weights saved in the .pt file
        print("Restoring model (that is the network\"s weights) from file...")

    model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                  "checkpoints",
                                  config.model.name,
                                  config.model.name + ".pt")))


    '''  INFERENCE  '''
    
    # Initialize the criterion (loss)
    criterion = nn.MSELoss()
    
    # Block for inferring the BC-Values
    if infer_BC:
        
        def process_sequence(model, criterion, data, batch_size, epoch, config,
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
        iterations = 350
        
        # Slice data for the inference
        u_infer = data[:data_infer,:]
        
        # Initialize lists to store
        left_BC = []
        right_BC = []
        
        left_BC_grad = []
        right_BC_grad = []
        
        mse_inference = []
        
        for i in range(iterations):
            #
            # Train the network on the given data
    
            # Define the closure function that consists of resetting the
            # gradient buffer, loss function calculation, and backpropagation
            # It is necessary for LBFGS optimizer, because it requires multiple
            # function evaluations
            def closure():
                # Set the model to train mode
                model.train()
                    
                # Reset the optimizer to clear data from previous iterations
                optimizer.zero_grad()
    
                _, mse = process_sequence(
                    model=model,
                    criterion=criterion,
                    data=u_infer,
                    batch_size=config.training.batch_size,
                    epoch=i,
                    config=config,
                    device=device
                )
                
                mse.backward()
                    
                return mse
            
            optimizer.step(closure)
                
            # Extract the MSE value from the closure function
            mse = closure()
            #
    
            # Append the error to the error list
            epoch_errors_train.append(mse.item())
    
            train_sign = "(-)"
            if epoch_errors_train[-1] < best_train:
                best_train = epoch_errors_train[-1]
                train_sign = "(+)"

            print(f"MSE: {train_sign}{mse} \n Iteration: {i}")
            mse_inference.append(float(mse))
            
            print(f"BC: {model.bc}")
            left_BC.append(float(model.bc[0,0,0]))
            right_BC.append(float(model.bc[0,0,1]))
            
            print(f"\n {model.bc.grad}")
            left_BC_grad.append(float(model.bc.grad[0,0,0]))
            right_BC_grad.append(float(model.bc.grad[0,0,1]))
            
        # Print the boundary conditions that are inferred by the model
        print(f"BC_final {model.bc}")
        
        # Cut the used values from the data
        data = data[data_infer:,:]
        t_series = t_series[data_infer:]
        
    """
    TESTING
    """
    
    model.eval()
    
    # Set up the training and validation datasets and -loaders
    data_test = th.clone(data) #th.tensor(data,device=device).unsqueeze(1)
    sequence_length = len(data_test) - 1

    # Evaluate the network for the given test data

    # Separate the data into network inputs and labels
    net_inputs = th.clone(data_test[:-1])
    net_labels = th.clone(data_test[1:])
    
    # Set up an array of zeros to store the network outputs
    net_outputs = th.zeros(size=(sequence_length,
                                 config.testing.batch_size,                                 
                                 config.model.dynamic_channels[-1],
                                 *config.model.field_size),
                           device=device)
    state_list = None

    # Iterate over the remaining sequence of the training example and perform a
    # forward pass
    time_start = time.time()
    for t in range(len(net_inputs)):

        if t == 0:
            # Initial input
            net_input = net_inputs[t]
        else:
            # Closed loop
            net_input = net_outputs[t - 1]

        net_output, state_list = model.forward(input_tensor=net_input,
                                               cur_state_list=state_list)
        
        net_outputs[t] = net_output

    if print_progress:
        forward_pass_duration = time.time() - time_start
        print("Forward pass took:", forward_pass_duration, "seconds.")

    # Convert the PyTorch network output tensor into a numpy array
    net_outputs = net_outputs.cpu().detach().numpy()[:, 0, 0]
    net_labels = net_labels.cpu().detach().numpy()[:, 0, 0]

    #
    # Visualize the data
    if visualize:
        if config.data.type == "burger" or\
           config.data.type == "allen_cahn":
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
        
            # u(t, x) over time
            fig, ax = plt.subplots()
               
            line1, = ax.plot(x, net_labels[0,:], 'b-', linewidth=2, label='Exact')
            line2, = ax.plot(x, net_outputs[0,:], 'ro', linewidth=2, label='Prediction')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$u(t,x)$')
            ax.set_xlim([x.min(), x.max()])
            ax.set_ylim([-2.1, 2.1])
       
            anim = animation.FuncAnimation(fig,
                                           animate_1d,
                                           frames=len(t_series) - 1,
                                           fargs=(line1, line2, np.transpose(net_labels), np.transpose(net_outputs)),
                                           interval=20)
           
        # Plot over space and time
        plt.style.use("dark_background")
             
        fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
     
        if config.data.type == "burger" or\
           config.data.type == "allen_cahn":
                 
            im1 = ax[0].imshow(
                np.transpose(net_labels), interpolation='nearest',
                cmap='rainbow', origin='lower', aspect='auto', vmin=-0.4,
                vmax=0.4
            )
            fig.colorbar(im1, ax=ax[0])
            im2 = ax[1].imshow(
                np.transpose(net_outputs), interpolation='nearest',
                cmap='rainbow', origin='lower', aspect='auto', vmin=-0.4,
                vmax=0.4
            )
            fig.colorbar(im2, ax=ax[1])
     
            ax[0].set_xlabel("t")
            ax[0].set_ylabel("x")
            ax[1].set_xlabel("t")
     
     
        ax[0].set_title("Ground Truth")
        ax[1].set_title("Network Output")
     
        plt.tight_layout()
        plt.draw()
        plt.show()

    #
    # Compute and return statistics
    mse = np.mean(np.square(net_outputs - net_labels))

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
    
    
def animate(t, im1, im2, net_labels, net_outputs):
    """
    Data animation function animating an image over time.
    :param t: The current time step
    :param axis: The matplotlib image object
    :param field: The data field
    :return: The matplotlib image object updated with the current time step's
        image date
    """
    im1.set_array(net_labels[t])
    im2.set_array(net_outputs[t])


if __name__ == "__main__":
    mse, inferred_BC = run_testing(print_progress=True, visualize=True)

    print(f"MSE_final: {mse}")
    print(f"inferred_BC: {inferred_BC}")

    print("Done.")