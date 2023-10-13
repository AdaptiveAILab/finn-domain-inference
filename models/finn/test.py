#! env/bin/python3

"""
Main file for testing (evaluating) or inferring a FINN model
"""

import numpy as np
import torch as th
import torch.nn as nn
import os
from threading import Thread
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import sys

sys.path.append("..")
from utils.configuration import Configuration
import utils.helper_functions as helpers
from finn import FINN_Burger, FINN_AllenCahn


def run_testing(visualize=True, model_number=None, infer_BC=False, infer_domain=False):
    # It is not possible to switch on only infer_domain without infer_BC
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

    if config.data.type == "burger":
        # Load samples, together with x and t series
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
        x = np.load(os.path.join(data_path, "x_series.npy"))
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                      dtype=th.float).to(device=device)
        print(u.shape)

        # Compute dx
        dx = x[1] - x[0]

        # Initialize and set up the model
        model = FINN_Burger(
            u=u,
            D=np.array([0.01/np.pi/dx ** 2]),
            BC=np.array([[0.0], [0.0]]),
            dx=dx,
            layer_sizes=config.model.layer_sizes,
            device=device,
            mode="test",
            learn_coeff=False
        ).to(device=device)

    elif config.data.type == "allen_cahn":
        # Load samples, together with x and t series
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
        x = np.load(os.path.join(data_path, "x_series.npy"))
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                      dtype=th.float).to(device=device)

        # Compute dx
        dx = x[1] - x[0]

        # Initialize and set up the model
        model = FINN_AllenCahn(
            u=u,
            D=np.array([0.005 / dx ** 2]),
            BC=np.array([[0.0], [0.0]]),
            dx=dx,
            layer_sizes=config.model.layer_sizes,
            device=device,
            mode="test",
            learn_coeff=True
        ).to(device=device)

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Trainable model parameters: {pytorch_total_params}\n")

    # Load the trained weights from the checkpoints into the model
    model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                               "checkpoints",
                                               config.model.name,
                                               config.model.name + ".pt")))

    # Initialize the criterion (loss)
    criterion = nn.MSELoss()

    '''  INFERENCE  '''

    if infer_BC:
        # Set all trainable parameters to false
        for p in model.parameters():
            p.requires_grad = False

        if config.data.type == "burger":
            # Set BC as parameter
            model.BC = nn.Parameter(th.tensor([[np.random.uniform(-1, 1)], [np.random.uniform(-1, 1)]], dtype=th.float))
            print(f"BC: {model.BC}")

        elif config.data.type == "allen_cahn":
            # Set BC as parameter
            model.BC = nn.Parameter(
                th.tensor([[np.random.uniform(-0.3, 0.3)], [np.random.uniform(-0.3, 0.3)]], dtype=th.float))
            print(f"BC: {model.BC}")

        # Count number of trainable parameters
        pytorch_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"Trainable model parameters for BCs: {pytorch_total_params}\n")

        if infer_domain:
            cut_domain_size = 10
            R = 10

            if R == 0:
                # Set the parts of the initial state as learnable parameters for domain reconstruction
                model.init_left = nn.Parameter(th.zeros(cut_domain_size, dtype=th.float)).to(device=device)
                model.init_right = nn.Parameter(th.zeros(cut_domain_size, dtype=th.float)).to(device=device)
            else:
                # Create retrospective reconstruction domain
                model.retro_domain = nn.Parameter(th.zeros((1, u.shape[-1]), dtype=th.float)).to(device=device)

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
        iterations = 10

        # Add noise to the data for the inference process
        u[:data_infer] = u[:data_infer] + th.normal(th.zeros_like(u[:data_infer]),
                                                    th.ones_like(u[:data_infer]) * config.inference.noise)

        # Slice data for the inference
        u_infer = u[:data_infer, :]

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
            # function evaluations.
            # It can be used for Adam as well.
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
                         u_infer[0, cut_domain_size:-cut_domain_size],
                         model.init_right)).unsqueeze(0)

                    # Forward propagation
                    u_hat_infer = model(t=t[:data_infer+R], u=model.retro_domain)

                    # Compute loss only for the inner "seen" domain omitting the retrospective inference
                    loss = criterion(u_hat_infer[R:, cut_domain_size:-cut_domain_size],
                                           u_infer[:, cut_domain_size:-cut_domain_size])
                else:
                    # Forward propagate and calculate loss function
                    u_hat_infer = model(t=t[:data_infer], u=u_infer)

                    loss = criterion(u_hat_infer, u_infer)

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
                        bc=np.array2string(model.BC.detach().numpy(), precision=5),
                        net=model,
                        infer=True))
                    thread.start()

            print(f"Loss: {infer_sign}{loss} \n Iteration: {i}")

            print(f"BC: {model.BC}")
            left_BC.append(float(model.BC[0, 0]))
            right_BC.append(float(model.BC[1, 0]))

            print(f"\n {model.BC.grad}")
            left_BC_grad.append(float(model.BC.grad[0, 0]))
            right_BC_grad.append(float(model.BC.grad[1, 0]))

            if infer_domain and R == 0:
                print(f"\n left: {model.init_left}")
                print(f"\n right: {model.init_right}")
        #### Inference loop is over ####


    '''Evaluation'''
    # Restore the model
    if infer_BC or infer_domain:
        print('Restoring the model')
        model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                                   "checkpoints",
                                                   'infer_' + config.model.name,
                                                   'infer_' + config.model.name + ".pt")))

    # Concatanate data and the inferred domain one last time for R=0
    if infer_domain:
        if R == 0:
            model.retro_domain = th.cat(
                (model.init_left,
                 u_infer[0, cut_domain_size:-cut_domain_size],
                 model.init_right)).unsqueeze(0)

    model.eval()

    # Print the inferred BC
    if infer_BC:
        print(f"BC_final: {model.BC}")

    # Forward data through the model
    if infer_domain:
        dt = 1 / (u.shape[0] - 1)

        t = th.tensor(np.linspace(0, (1.0 + (R * dt)), u.shape[0] + R), dtype=th.float).to(device=device)
        u_hat = model(t=t, u=model.retro_domain).detach().cpu()

        # Save certain inferred retrospective domains
        # In case R=10
        # inferred_retro_domains = {
        #     # model.retro_domain is the same as u_hat[0]
        #     f'(infer_{config.model.name}: t={-R})':   [model.retro_domain.detach().numpy()],
        #     f'(t={-R + 1})': [u_hat[1]],
        #     f'(t={-R + 2})': [u_hat[2]],
        #     f'(t={-R + 3})': [u_hat[3]],
        #     f'(t={-R + 4})': [u_hat[4]],
        #     f'(t={-R + 5})': [u_hat[5]],
        #     f'(t={-R + 6})': [u_hat[6]],
        #     f'(t={-R + 7})': [u_hat[7]],
        #     f'(t={-R + 8})': [u_hat[8]],
        #     f'(t={-R + 9})': [u_hat[9]],
        #     f'(t={-R+R})': [u_hat[R]]
        # }

        # In case R=5
        # inferred_retro_domains = {
        #     # model.retro_domain is the same as u_hat[0]
        #     f'(infer_{config.model.name}: t={-R})':   [model.retro_domain.detach().numpy()],
        #     f'(t={-R + 1})': [u_hat[1]],
        #     f'(t={-R + 2})': [u_hat[2]],
        #     f'(t={-R + 3})': [u_hat[3]],
        #     f'(t={-R + 4})': [u_hat[4]],
        #     f'(t={-R + 5})': [u_hat[R]]
        # }

        # In case R=2
        # inferred_retro_domains = {
        #     # model.retro_domain is the same as u_hat[0]
        #     f'(infer_{config.model.name}: t={-R})': [model.retro_domain.detach().numpy()],
        #     f'(t={-R + 1})': [u_hat[1]],
        #     f'(t={-R + 2})': [u_hat[R]]
        # }

        # In case R=1
        # inferred_retro_domains = {
        #     # model.retro_domain is the same as u_hat[0]
        #     f'(infer_{config.model.name}: t={-R})': [model.retro_domain.detach().numpy()],
        #     f'(t={-R + 1})': [u_hat[R]],
        # }

        # In case R=0
        # inferred_retro_domains = {
        #     f'(infer_{config.model.name}: t = {-R + R})': [model.retro_domain.detach().numpy()]
        # }

        # Print the inference at t = 0
        inferred_retro_domains = {
            f'(infer_{config.model.name}: t = {-R + R})': [u_hat[R]]
        }
        print(f"inferred_retro_domains: {inferred_retro_domains}")

        # Discard the past
        t = th.tensor(np.linspace(0, 1.0, u.shape[0]), dtype=th.float).to(device=device)
        u_hat = u_hat[R:,:]
    else:
        if infer_BC:
            u_hat = model(t=t[data_infer:], u=u[data_infer:]).detach().cpu()
        else:
            u_hat = model(t=t, u=u).detach().cpu()

    # Put u and t on CPU
    u = u.cpu()
    t = t.cpu()

    # Compute test error
    if infer_domain:
        # Exclude the inference data from the error computation
        # Whole-Domain Prediction Error
        # mse = criterion(u_hat[data_infer:, :],
        #                 u[data_infer:, :]).item()

        # Seen-Domain Prediction Error
        mse = criterion(u_hat[data_infer:, cut_domain_size:-cut_domain_size],
                        u[data_infer:, cut_domain_size:-cut_domain_size]).item()
    else:
        if infer_BC:
            mse = criterion(u_hat, u[data_infer:]).item()
        else:
            mse = criterion(u_hat, u).item()
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
                # Choose hline according to the data BC
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

            # u(t, x) over space
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))

            if not infer_domain and infer_BC:
                ax[0].imshow(np.flip(np.concatenate((u[:data_infer].numpy().copy(),
                                                     u_hat.numpy().copy())), 0),
                             interpolation='nearest',
                             extent=[x.min(), x.max(),
                                     t.min(), t.max()],
                             origin='upper', aspect='auto')

                ax[0].axhline(y=t[data_infer], color='red')
            else:
                ax[0].imshow(np.flip(u_hat.numpy().copy(), 0),
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
            ax[0].set_title('FINN', fontsize=26)
            ax[0].grid(False)

            h = ax[1].imshow(np.flip(u.numpy().copy(), 0),
                             interpolation='nearest',
                             extent=[x.min(), x.max(),
                                     t.min(), t.max()],
                             origin='upper', aspect='auto')
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax, ticks=[-1.0, -0.5, 0, 0.5, 1.0])
            cbar.ax.set_yticklabels(['-1.0', '-0.5', '0', '0.5', '1.0'])

            if config.inference.noise > 0.0 and infer_domain:
                ax[1].axhline(y=t[data_infer], color='red')
            ax[1].set_ylim(t.min(), t.max())
            ax[1].set_xlim(x.min(), x.max())
            ax[1].set_xticks([x.min(), 0, x.max()])
            ax[1].set_xticklabels([f'{x.min()}', '0', f'{x.max()}'])
            ax[1].set_yticks([t.min(),0.5, t.max()])
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
                u = u[:, data_infer:]
                t = t[data_infer:]

            fig, ax = plt.subplots()
            line1, = ax.plot(x, u[:, 0], 'ro', linewidth=2, label='Data')
            line2, = ax.plot(x, u_hat[:, 0], 'b-', linewidth=2, label='Prediction')
            ax.set_xlabel('$x$', fontsize=24)
            ax.set_ylabel('$u(t,x)$', fontsize=24)
            ax.set_xlim([x.min(), x.max()])
            # Set ylim according to the data BC
            ax.set_ylim([-1.4, 1.4])
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

            # Save the animation
            # f = f"{config.model.name}_animation.mp4"
            # writervideo = animation.FFMpegWriter(fps=60)
            # anim.save(f, writer=writervideo)

            # u(t,x) at one specific t
            fig, ax = plt.subplots(sharex=True, figsize=(6, 4))
            ax.plot(x, u[:, -1], 'ro', linewidth=2.0) #, label='Exact')
            ax.plot(x, u_hat[:, -1], 'b-', linewidth=2.0) #, label='Prediction')

            ax.set_xlabel('$x$', fontsize=24)
            ax.set_ylabel(f'$u(x, t={int(t.max())})$', fontsize=24)
            # Choose lims and ticks according to the data
            # ax.set_xlim([x.min(), x.max()])
            # ax.set_ylim([-1.7, 1.7])
            ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            # ax.set_xticks([x.min(), 0, x.max()])
            #ax.legend(loc="upper left", fontsize=16)
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
        return mse, model.BC, inferred_retro_domains
    else:
        return mse, model.BC


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