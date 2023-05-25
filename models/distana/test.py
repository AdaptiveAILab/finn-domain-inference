import numpy as np
import torch as th
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
from convlstm import ConvLSTM


def run_testing(print_progress=False, visualize=False, model_number=None):

    th.set_num_threads(1)
    
    # Load the user configurations
    config = Configuration("config.json")

    # Append the model number to the name of the model
    if model_number is None:
        model_number = config.model.number
    config.model.name = config.model.name + "_" + str(model_number).zfill(2)

    # Hide the GPU(s) in case the user specified to use the CPU in the config
    # file
    if config.general.device == "CPU":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Set device on GPU if specified in the configuration file, else CPU
    device = "cpu" #helpers.determine_device()

    # Initialize and set up the network
    model = DISTANA(config=config, device=device).to(device=device)
    # model = ConvLSTM(input_dim=1,
    #                  hidden_dim=[32, 64, 32, 16, 1],
    #                  kernel_size=(3, 3),
    #                  num_layers=5,
    #                  batch_first=True,
    #                  bias=True,
    #                  return_all_layers=False)

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
    model.eval()

    """
    TESTING
    """

    #
    # Load the data depending on the task
    if config.data.type == "burger":
        data_path = os.path.join("../../data/",
                                 config.data.type,
                                 config.data.name,
                                 "sample.npy")
        data = np.array(np.load(data_path), dtype=np.float32)
        data = np.expand_dims(data, axis=1)

    elif config.data.type == "diffusion_sorption":        
        data_path_base = os.path.join("../../data/",
                                      config.data.type,
                                      config.data.name)
        data_path_c = os.path.join(data_path_base, "sample_c.npy")
        data_path_ct = os.path.join(data_path_base, "sample_ct.npy")
        data_c = np.array(np.load(data_path_c), dtype=np.float32)
        data_ct = np.array(np.load(data_path_ct), dtype=np.float32)
        data = np.stack((data_c, data_ct), axis=1)

    elif config.data.type == "diffusion_reaction":
        data_path_base = os.path.join("../../data/",
                                      config.data.type,
                                      config.data.name)
        data_path_u = os.path.join(data_path_base, "sample_u.npy")
        data_path_v = os.path.join(data_path_base, "sample_v.npy")
        data_u = np.array(np.load(data_path_u), dtype=np.float32)
        data_v = np.array(np.load(data_path_v), dtype=np.float32)
        data = np.stack((data_u, data_v), axis=1)
        
    elif config.data.type == "allen_cahn":
        data_path = os.path.join("../../data/",
                                 config.data.type,
                                 config.data.name,
                                 "sample.npy")
        data = np.array(np.load(data_path), dtype=np.float32)
        data = np.expand_dims(data, axis=1)

    elif config.data.type == "burger_2d":
        data_path = os.path.join("../../data/",
                                 config.data.type,
                                 config.data.name,
                                 "sample.npy")
        data = np.array(np.load(data_path), dtype=np.float32)
        data = np.expand_dims(data, axis=1)

    elif config.data.type == "shallow_water":
        import h5py

        # Load the data
        root_path = os.path.abspath("../../data")
        data_path = os.path.join(root_path, config.data.type, config.data.name)
        f = h5py.File(os.path.join(data_path, "wave_rotation.hdf5"), 'r')
        print(list(f.keys()))

        dset = f['train']['labels']

        # Transform the data into a tensor
        data = th.tensor(dset, dtype=th.float).to(device=device)

        # Just take one sample from the whole data
        data = data[0].unsqueeze(1)
        print(f"data -> {data.shape}")

        # ---------------- Set specific parameters depending on the data ---------------- #
        Lx = 1E+2  # Length of domain in x-direction
        Ly = 1E+2  # Length of domain in y-direction
        Nx = data.shape[-2]  # Number of grid points in x-direction
        Ny = data.shape[-1]  # Number of grid points in y-direction
        dx = Lx / (Nx - 1)  # Grid spacing in x-direction
        dy = Ly / (Ny - 1)  # Grid spacing in y-direction

        g = 9.81  # Acceleration of gravity [m/s^2]
        H = 10  # Depth of fluid [m] if two dimensional array, you can decide where shallow where deep

        x = np.linspace(-Lx / 2, Lx / 2, Nx)  # Array with x-points
        y = np.linspace(-Ly / 2, Ly / 2, Ny)  # Array with y-points

        # Generate t in order to make the data compatible for FINN
        th.set_printoptions(precision=10)
        dt = 0.1 * min(dx, dy) / np.sqrt(g * H)  # Time step (defined from the CFL condition)
        sample_interval = 1
        t_steps = data.shape[0]

        t_temp = th.zeros(t_steps * sample_interval, dtype=th.float64, device=device)

        for i in range(1, t_steps * sample_interval):
            t_temp[i] += t_temp[i - 1] + dt

        t_series = th.zeros(t_steps, dtype=th.float64, device=device)
        counter = 1

        for i in range(1, t_steps * sample_interval):
            if i % sample_interval == 0:
                t_series[counter] = t_temp[i]
                counter += 1

    # Set up the test set and -loaders
    data_test = th.clone(data).unsqueeze(1).to(device=device) #th.tensor(data, device=device)#.unsqueeze(1)
    sequence_length = len(data_test) - 1

    # Evaluate the network for the given test data

    # Separate the data into network inputs and labels
    net_inputs = th.clone(data_test[:-1])
    net_labels = th.clone(data_test[1:])

    # Test DISTANA
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
        if t < config.testing.teacher_forcing_steps:
            # Teacher forcing
            net_input = net_inputs[t]
        else:
            # Closed loop
            net_input = net_outputs[t - 1]

        # Feed the boundary data also in closed loop if desired
        if config.testing.feed_boundary_data:
            net_input[:, :, 0] = net_inputs[t, :, :, 0]
            net_input[:, :, -1] = net_inputs[t, :, :, -1]

        net_output, state_list = model.forward(input_tensor=net_input,
                                               cur_state_list=state_list)
        net_outputs[t] = net_output

    if print_progress:
        forward_pass_duration = time.time() - time_start
        print("Forward pass took:", forward_pass_duration, "seconds.")

    # Convert the PyTorch network output tensor into a numpy array
    net_outputs = net_outputs.squeeze().cpu().detach().numpy()
    net_labels  = net_labels.squeeze().cpu().detach().numpy()

    # # Test ConvLSTM
    # net_outputs = model(net_inputs)
    # # Convert the PyTorch network output tensor into a numpy array
    # net_outputs = net_outputs[0][0].squeeze().cpu().detach().numpy() #net_outputs.cpu().detach().numpy()[:, 0, 0]
    # net_labels = net_labels.squeeze().cpu().detach().numpy() #[:, 0, 0]

    #
    # Visualize the data
    if visualize:
        # plt.style.use("dark_background")

        # Plot over space and time
        # fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        if config.data.type == "burger" or\
           config.data.type == "diffusion_sorption" or\
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

        elif config.data.type == "diffusion_reaction":
            
            im1 = ax[0].imshow(
                np.transpose(net_labels[..., 0]), interpolation='nearest',
                cmap='rainbow', origin='lower', aspect='auto', vmin=-0.4,
                vmax=0.4
            )
            fig.colorbar(im1, ax=ax[0])
            im2 = ax[1].imshow(
                np.transpose(net_outputs[..., 0]), interpolation='nearest',
                cmap='rainbow', origin='lower', aspect='auto', vmin=-0.4,
                vmax=0.4
            )
            fig.colorbar(im2, ax=ax[1])

            ax[0].set_xlabel("x")
            ax[0].set_ylabel("y")
            ax[1].set_xlabel("x")

        elif config.data.type == "shallow_water":
            u_hat = net_outputs
            u = net_labels #data[:, 0, :].detach().cpu().numpy()
            timesteps = len(t_series)

            # Plot the wave activity at one position
            fig, ax = plt.subplots(1, 2, figsize=[8, 2])

            ax[0].plot(range(len(u)), u[:, 12, 12])
            ax[0].set_xlabel("Time")
            ax[0].set_ylabel("Wave amplitude")
            ax[0].set_title("Ground Truth")
            ax[0].set_xlim([0, timesteps - 1])

            ax[1].plot(range(len(u_hat)), u_hat[:, 12, 12])  # , 0])
            ax[1].set_xlabel("Time")
            ax[1].set_ylabel("Wave amplitude")
            ax[1].set_title("Prediction")
            ax[1].set_xlim([0, timesteps - 1])

            plt.tight_layout()
            plt.show()

            # Animate the spatio-temporal wave
            fig, ax = plt.subplots(1, 2, figsize=(6, 6))

            im1 = ax[0].imshow(u_hat[0, :, :], interpolation='nearest', vmin=-0.6, vmax=0.6, cmap="Blues")
            ax[0].set_title("Prediction")

            im2 = ax[1].imshow(u[0, :, :], interpolation='nearest', vmin=-0.6, vmax=0.6, cmap="Blues")
            ax[1].set_title("Ground Truth")

            anim = animation.FuncAnimation(fig,
                                           animate_2d,
                                           frames=timesteps-1,
                                           fargs=(im1, im2, u_hat, u),
                                           interval=10)
            plt.tight_layout()
            plt.draw()
            plt.show()

            # # Save the animation
            # f = f"{config.model.name}_animation.mp4"
            # writervideo = animation.FFMpegWriter(fps=60)
            # anim.save(f, writer=writervideo)

        elif config.data.type == "burger_2d":
            
            im1 = ax[0].imshow(
                np.transpose(net_labels[-1, ...]), interpolation='nearest',
                cmap='rainbow', origin='lower', aspect='auto', vmin=-0.4,
                vmax=0.4
            )
            fig.colorbar(im1, ax=ax[0])
            im2 = ax[1].imshow(
                np.transpose(net_outputs[-1, ...]), interpolation='nearest',
                cmap='rainbow', origin='lower', aspect='auto', vmin=-0.4,
                vmax=0.4
            )
            fig.colorbar(im2, ax=ax[1])

            ax[0].set_xlabel("x")
            ax[0].set_ylabel("y")
            ax[1].set_xlabel("x")


        ax[0].set_title("Ground Truth")
        ax[1].set_title("Network Output")


        if config.data.type == "diffusion_reaction"\
            or config.data.type == "burger_2d":
            anim = animation.FuncAnimation(
                fig,
                animate,
                frames=sequence_length,
                fargs=(im1, im2, net_labels, net_outputs),
                interval=20
            )

        plt.show()


    #
    # Compute and return statistics
    mse = np.mean(np.square(net_outputs - net_labels))

    return mse.item(), #net_outputs, net_labels


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

def animate_2d(t, im1, im2, u_hat, u):
    """
    Data animation function animating an image over time.
    :param t: The current time step
    :param axis: The matplotlib image object
    :param field: The data field
    :return: The matplotlib image object updated with the current time step's
        image date
    """
    im1.set_array(u_hat[t, :, :])
    im2.set_array(u[t, :, :])


if __name__ == "__main__":
    mse = run_testing(print_progress=True, visualize=True)

    print(f"MSE: {mse}")
    print("Done.")