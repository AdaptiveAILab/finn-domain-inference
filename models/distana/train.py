import numpy as np
import torch as th
import torch.nn as nn
import time
import glob
import os
import matplotlib.pyplot as plt
from threading import Thread
import sys

sys.path.append("..")
from utils.configuration import Configuration
import utils.helper_functions as helpers
from distana import DISTANA


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

    # Probability of closed loop in current epoch
    p_cl = epoch / (config.training.epochs * 1.5)

    # Iterate over the whole sequence of the training example and
    # perform a forward pass
    for t in range(1, len(net_inputs)):

        # Draw random number to determine teacher forcing or closed loop
        # (scheduled sampling)
        p_uniform = np.random.uniform(0, 1)
        closed_loop = True if p_uniform < p_cl else False

        if not closed_loop:
            # Teacher forcing
            net_input = net_inputs[t]
        else:
            # Closed loop
            net_input = net_outputs[t - 1]

        # Forward the input through the network
        net_output, state_list = model.forward(input_tensor=net_input,
                                               cur_state_list=state_list)
        
        # Store the output of the network for this sequence step
        net_outputs[t] = net_output
        

    # Compute the mean squared error
    mse = criterion(net_outputs, net_labels)

    return net_outputs, mse


def run_training(print_progress=True, model_number=None):
    
    # decide if mini-batch training or not
    train_multi_batch = False

    # Set a random seed for varying weight initializations
    th.seed()

    th.set_num_threads(1)
    
    # Load the user configurations
    config = Configuration("config.json")
    
    # setting device on GPU if available, else CPU
    device = config.general.device

    # Append the model number to the name of the model
    if model_number is None:
        model_number = config.model.number
    config.model.name = config.model.name + "_" + str(model_number).zfill(2)
    
    print("Model name:", config.model.name)

    # Hide the GPU(s) in case the user specified to use the CPU in the config file
    if config.general.device == "CPU":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    #
    # Load the data depending on the task
    if config.data.type == "burger":
        if train_multi_batch:
            
            root_path = os.path.abspath("../../data")
            data_path = os.path.join(root_path, config.data.type, config.data.name)
            
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
                    data = u.unsqueeze(1).unsqueeze(1)
                    
                    bc = th.tensor([[[left_BC[shuffle[idx]], right_BC[shuffle[idx]]]]])
                else:
                    data = th.cat((data, u.unsqueeze(1).unsqueeze(1)), dim=1).to(device=device)
                    
                    bc = th.cat((bc, th.tensor([[[left_BC[shuffle[idx]], right_BC[shuffle[idx]]]]])), dim=0)
                print(data.shape)
                print(bc.shape)
        else:
            data_path = os.path.join("../../data/",
                                     config.data.type,
                                     config.data.name,
                                     "sample.npy")
            data = np.array(np.load(data_path), dtype=np.float32)
            data = np.expand_dims(data, axis=1)

            # Uncomment below if you want to train a shorter sequence
            # (adjust valid and train data accordingly)
            # data = data[:30]
            # print(data.shape)
            
            bc = th.tensor([[[0.5, -0.5]]], device=device)
        
    elif config.data.type == "allen_cahn":
        if train_multi_batch:
            
            root_path = os.path.abspath("../../data")
            data_path = os.path.join(root_path, config.data.type, config.data.name)
            
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
                    data = u.unsqueeze(1).unsqueeze(1)
                    
                    bc = th.tensor([[[left_BC[shuffle[idx]], right_BC[shuffle[idx]]]]])
                else:
                    data = th.cat((data, u.unsqueeze(1).unsqueeze(1)), dim=1).to(device=device)
                    
                    bc = th.cat((bc, th.tensor([[[left_BC[shuffle[idx]], right_BC[shuffle[idx]]]]])), dim=0)
                print(data.shape)
                print(bc.shape)
        else:
            data_path = os.path.join("../../data/",
                                     config.data.type,
                                     config.data.name,
                                     "sample.npy")
            data = np.array(np.load(data_path), dtype=np.float32)
            data = np.expand_dims(data, axis=1)

            # Uncomment below if you want to train a shorter sequence
            # (adjust valid and train data accordingly)
            # data = data[:30]
            # print(data.shape)
            
            bc = th.tensor([[[-0.5, 0.5]]], device=device)

    #
    # Set up the training and validation datasets and -loaders
    if not train_multi_batch:
        data_train = th.tensor(
            data[:config.training.t_stop],
            device=device
        ).unsqueeze(1)
        data_valid = th.tensor(
            data[config.validation.t_start:config.validation.t_stop],
            device=device
        ).unsqueeze(1)
    else:
        data_train = data[:config.training.t_stop]
        data_valid = data[config.validation.t_start:config.validation.t_stop]


    # Initialize and set up the network
    model = DISTANA(config=config, device=device, bc=bc, learn_BC=False).to(device=device)
    
    if model.learn_BC:
        # Initialize lists to store
        left_BC = []
        right_BC = []
        
        left_BC_grad = []
        right_BC_grad = []
        
        mse_train = []

    if print_progress:
        # Count number of trainable parameters
        pytorch_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print("Trainable model parameters:", pytorch_total_params)
        
    # If desired, restore the network by loading the weights saved in the .pt file
    if config.training.continue_training:
        print("Restoring model (that is the network\"s weights) from file...")
        model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                      "checkpoints",
                                      config.model.name,
                                      config.model.name + ".pt")))
        model.train()

    #
    # Set up the optimizer and the criterion (loss)
    optimizer = th.optim.LBFGS(model.parameters(),
                                lr=config.training.learning_rate)
    
    criterion = nn.MSELoss()

    #
    # Set up lists to save and store the epoch errors
    epoch_errors_train = []
    epoch_errors_valid = []
    best_train = np.infty
    best_valid = np.infty
    
    """
    TRAINING
    """

    a = time.time()

    #
    # Start the training and iterate over all epochs
    for epoch in range(config.training.epochs):

        epoch_start_time = time.time()

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
                data=data_train,
                batch_size=config.training.batch_size,
                epoch=epoch,
                config=config,
                device=device
            )
            
            mse.backward()
            
            if model.learn_BC:
                print(f"BC: {model.bc}")
                left_BC.append(float(model.bc[0,0,0]))
                right_BC.append(float(model.bc[0,0,1]))
                
                print(f"\n BC_grad: {model.bc.grad}")
                left_BC_grad.append(float(model.bc.grad[0,0,0]))
                right_BC_grad.append(float(model.bc.grad[0,0,1]))
                
                print(f"mse_train: {mse.item()}")
                mse_train.append(float(mse))
                
            return mse
        
        optimizer.step(closure)
            
        # Extract the MSE value from the closure function
        mse = closure()

        # Append the error to the error list
        epoch_errors_train.append(mse.item())

        train_sign = "(-)"
        if epoch_errors_train[-1] < best_train:
            best_train = epoch_errors_train[-1]
            train_sign = "(+)"

        #
        # Validate the network
        net_output, mse = process_sequence(
            model=model,
            criterion=criterion,
            data=data_valid,
            batch_size=config.training.batch_size,
            epoch=epoch,
            config=config,
            device=device
        )
        epoch_errors_valid.append(mse.item())

        # Create a plus or minus sign for the validation error
        valid_sign = "(-)"
        if epoch_errors_valid[-1] < best_valid:
            best_valid = epoch_errors_valid[-1]
            valid_sign = "(+)"

            if config.training.save_model:                
                # Start a separate thread to save the model
                thread = Thread(target=helpers.save_model_to_file(
                    model_src_path=os.path.abspath(""),
                    config=config,
                    epoch=epoch,
                    epoch_errors_train=epoch_errors_train,
                    epoch_errors_valid=epoch_errors_valid,
                    bc=np.array2string(model.bc.detach().numpy(), precision=5),
                    net=model))
                thread.start()

        #
        # Print progress to the console
        if print_progress:
            print(f"Epoch {str(epoch+1).zfill(int(np.log10(config.training.epochs))+1)}/{str(config.training.epochs)} took {str(np.round(time.time() - epoch_start_time, 2)).ljust(5, '0')} seconds. \t\tAverage epoch training error: {train_sign}{str(np.round(epoch_errors_train[-1], 10)).ljust(12, ' ')} \t\tValidation error: {valid_sign}{str(np.round(epoch_errors_valid[-1], 10)).ljust(12, ' ')}")
        
    b = time.time()
    if print_progress:
        print("\nTraining took " + str(np.round(b - a, 2)) + " seconds.\n\n")
        
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

    return best_train

if __name__ == "__main__":
    mse, learnt_bc = run_training(print_progress=True)

    print("Done.")