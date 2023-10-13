"""
This script creates data using Burger's equation.
"""

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from simulator import Simulator


##############
# PARAMETERS #
##############

TRAIN_DATA = False
MINI_BATCHES = False

#
# Burger's specifications
DIFFUSION_COEFFICIENT = 0.01/np.pi

if TRAIN_DATA:
    T_MAX = 2
    T_STEPS = 256
    LEFT_BC = 3.0
    RIGHT_BC = -3.0
    print("This is training data")
else:
    T_MAX = 1
    T_STEPS = 128
    LEFT_BC = 1.5 #np.random.uniform(-1,1)
    RIGHT_BC = -1.5 #np.random.uniform(-1,1)
    print(LEFT_BC)
    print(RIGHT_BC)
    
X_LEFT = -1.0
X_RIGHT = 1.0
X_STEPS = 51

#
# Dataset
DATAPOINTS_INITIAL = 15
DATAPOINTS_BOUNDARY = 50
DATAPOINTS_COLLOCATION = 7500
DATASET_NAME = "data"
SAVE_DATA = True
VISUALIZE_DATA = True


#############
# FUNCTIONS #
#############

def generate_sample(simulator, visualize, save_data, root_path, idx):
    """
    This function generates a data sample, visualizes it if desired and saves
    the data to file if desired.
    :param simulator: The simulator object for data creation
    :param visualize: Boolean indicating whether to visualize the data
    :param save_data: Boolean indicating whether to write the data to file
    :param root_path: The root path of this script
    :param idx: Index for creating different sequences
    """
    
    print("Generating data...")
    # Generate a data sample
    sample = simulator.generate_sample()
    
    if TRAIN_DATA:
        
        # Randomly draw indices for initial, boundary and collocation points
        idcs_init, idcs_bound = draw_indices(
            simulator=simulator,
            n_init=DATAPOINTS_INITIAL,
            n_bound=DATAPOINTS_BOUNDARY,
            n_colloc=DATAPOINTS_COLLOCATION
        )
    
        # If specified, visualize the sample
        if visualize:
            visualize_sample(sample=sample,
                             simulator=simulator,
                             idcs_init=idcs_init,
                             idcs_bound=idcs_bound)
    
        # If specified, write the sample to file
        if save_data:
            
            write_data_to_file(
                root_path=root_path,
                simulator=simulator,
                sample=sample,
                idx=idx
            )
        
            # List for tuples as train/val/test data
            data_tuples = []
                
            # Concatenate all indices and add their data tuples to the list
            all_idcs = np.concatenate((idcs_init, idcs_bound), axis=0)
            for pair in all_idcs:
                data_tuples.append(create_data_tuple_init_bound(
                    sample=sample, pair=pair, simulator=simulator
                ))
            data_tuples.extend(np.transpose(create_data_tuple_colloc(
                sample=sample, simulator=simulator
            )))
            
        
            write_tuples_to_file(root_path, data_tuples, mode="train")
        
        # Training data (validation)
        
        # Randomly draw indices for initial, boundary and collocation points
        idcs_init, idcs_bound = draw_indices(
            simulator=simulator,
            n_init=DATAPOINTS_INITIAL,
            n_bound=DATAPOINTS_BOUNDARY,
            n_colloc=DATAPOINTS_COLLOCATION
        )
        
        # If specified, write the sample to file
        if save_data:
            
            # List for tuples as train/val/test data
            data_tuples = []
                
            # Concatenate all indices and add their data tuples to the list
            all_idcs = np.concatenate((idcs_init, idcs_bound), axis=0)
            
            for pair in all_idcs:
                data_tuples.append(create_data_tuple_init_bound(
                    sample=sample, pair=pair, simulator=simulator
                ))
            data_tuples.extend(np.transpose(create_data_tuple_colloc(
                sample=sample, simulator=simulator
            )))
        
            write_tuples_to_file(root_path, data_tuples, mode="val")
        
    
    else:
        
        # If specified, visualize the sample
        if visualize:
            visualize_sample(sample=sample,
                                simulator=simulator)
            
        # If specified, write the sample to file
        if save_data:
        
            write_data_to_file(
                root_path=root_path,
                simulator=simulator,
                sample=sample,
                idx=42 #it is a dummy number
            )
        


def draw_indices(simulator, n_init, n_bound, n_colloc):
    """
    Randomly chooses a specified number of points from the spatiotemporal
    sample for the initial and boundary conditions as well as collocation
    points.
    :param simulator: The simulator that created the sample
    :param n_init: Number of initial points at t=0
    :param n_bound: Number of boundary points at x_left and x_right
    :param n_colloc: Number of collocation points
    :return: The two-dimensional index arrays(t, x)
    """

    rng = np.random.default_rng()

    idcs_init = np.zeros((n_init, 2), dtype=np.int)
    idcs_init[:, 0] = 0
    idcs_init[:, 1] = rng.choice(len(simulator.x),
                                 size=n_init,
                                 replace=False)

    idcs_bound = np.zeros((n_bound, 2), dtype=np.int)
    idcs_bound[:n_bound//2, 0] = rng.choice(len(simulator.t)//2 + 1,
                                  size=n_bound//2,
                                  replace=False)
    idcs_bound[:n_bound//2, 1] = 0
    
    idcs_bound[n_bound//2:, 0] = rng.choice(len(simulator.t)//2 + 1,
                                  size=n_bound - n_bound//2,
                                  replace=False)
    idcs_bound[n_bound//2:, 1] = len(simulator.x) - 1

    return idcs_init, idcs_bound


def write_data_to_file(root_path, simulator, sample, idx):
    """
    Writes the given data to the according directory in .npy format.
    :param root_path: The root_path of the script
    :param simulator: The simulator that created the data
    :param sample: The sample to be written to file
    :param idx: Index for creating different sequences
    """
    
    if TRAIN_DATA:
    
        # Create the data directory for the training data if it does not yet exist
        data_path = os.path.join(root_path, DATASET_NAME+"_train")
        os.makedirs(data_path, exist_ok=True)
        
        # Write the t- and x-series data along with the sample to file
        if MINI_BATCHES:
            np.save(file=os.path.join(data_path, "t_series_mini_batches.npy"),
                arr=simulator.t[:len(simulator.t)])
            
            np.save(file=os.path.join(data_path, f"sample_{str(idx).zfill(2)}.npy"),
                    arr=sample[:len(simulator.t)])
        
        else:
            np.save(file=os.path.join(data_path, "t_series.npy"),
                    arr=simulator.t[:len(simulator.t)])
            
            np.save(file=os.path.join(data_path, "sample.npy"),
                    arr=sample[:len(simulator.t)])
            
        np.save(file=os.path.join(data_path, "x_series.npy"), arr=simulator.x)
            
        # Create the data directory for the extrapolation data if it does not yet exist
        data_path = os.path.join(root_path, DATASET_NAME+"_ext")
        os.makedirs(data_path, exist_ok=True)
        
        # Write the t- and x-series data along with the sample to file
        np.save(file=os.path.join(data_path, "t_series.npy"), arr=simulator.t)
        np.save(file=os.path.join(data_path, "x_series.npy"), arr=simulator.x)
        np.save(file=os.path.join(data_path, "sample.npy"), arr=sample)
    
    else:

        # Create the data directory if it does not yet exist
        data_path = os.path.join(root_path, DATASET_NAME+"_test")
        os.makedirs(data_path, exist_ok=True)
    
        # Write the t- and x-series data along with the sample to file
        np.save(file=os.path.join(data_path, "t_series.npy"), arr=simulator.t)
        np.save(file=os.path.join(data_path, "x_series.npy"), arr=simulator.x)
        np.save(file=os.path.join(data_path, "sample.npy"), arr=sample)
    
    


def write_tuples_to_file(root_path, data_tuples, mode):
    """
    Writes the data tuples to the according directory in .npy format for
    training and validation of PINN
    :param root_path: The root_path of the script
    :param data_tuples: Array of the train/val tuples
    :param mode: Any of "train" or "val"
    
    """
    
    data_path = os.path.join(root_path, DATASET_NAME+"_train")
    os.makedirs(os.path.join(data_path, mode), exist_ok=True)
    
    # Iterate over the data_tuples and write them to separate files
    for idx, data_tuple in enumerate(data_tuples):
        
        name = f"{mode}_{str(idx).zfill(5)}.npy"
        np.save(file=os.path.join(data_path, mode, name), arr=data_tuple)


def create_data_tuple_init_bound(simulator, sample, pair):
    """
    Creates a tuple (t, x, sample, t_idx, x_idx), where t is the
    time step, x is the spatial coordinate, sample is the desired model output,
    and t_idx and x_idx are the indices in the sample for t and x.
    :param simulator: The simulator that generated the sample
    :param sample: The data sample
    
    :param pair: The index pair of the current data points
    :return: Tuple (t, x, sample, t_idx, x_idx)
    """
    t_idx, x_idx = pair
    u = sample[t_idx, x_idx]
    t, x = simulator.t[t_idx], simulator.x[x_idx]
    
    return np.array((t, x, u, t_idx, x_idx), dtype=np.float32)

def create_data_tuple_colloc(simulator, sample):
    """
    Creates a tuple (t, x, sample, t_idx, x_idx), where t is the
    time step, x is the spatial coordinate, sample is the desired model output,
    and t_idx and x_idx are the indices in the sample for t and x.
    :param simulator: The simulator that generated the sample
    :param sample: The data sample
    
    :param pair: The index pair of the current data points
    :return: Tuple (t, x, sample, t_idx, x_idx)
    """
    t = np.arange(len(simulator.t)//2 + 1)
    x = np.arange(len(simulator.x))
    
    t, x = np.meshgrid(t,x)
    
    pair = np.hstack((t.flatten()[:,None],x.flatten()[:,None]))
    idx = np.random.choice(len(pair), DATAPOINTS_COLLOCATION , replace=True)
    
    t_idx = pair[idx,0]
    x_idx = pair[idx,1]
    
    u = sample[t_idx, x_idx]
    
    t, x = simulator.t[t_idx], simulator.x[x_idx]
    
    return np.array((t, x, u, t_idx, x_idx), dtype=np.float32)


def visualize_sample(sample, simulator, idcs_init=None, idcs_bound=None):
    """
    Method to visualize a single sample. Code taken and modified from
    https://github.com/maziarraissi/PINNs
    :param sample: The actual data sample for visualization
    :param simulator: The simulator used for data generation
    :param idcs_init: The indices of the initial points
    :param idcs_bound: The indices of the boundary points
    """

    ### This part saves the plot of the data ###
    # u(t, x) over time and space
    fig, ax = plt.subplots()

    h = ax.imshow(np.flip(sample, 0), interpolation='nearest',
                  extent=[simulator.x.min(), simulator.x.max(),
                          simulator.t.min(), simulator.t.max()],
                  origin='upper', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = fig.colorbar(h, cax=cax, ticks=[-0.99, -0.5, 0, 0.5, 0.99])
    cbar.ax.set_yticklabels(['-0.99', '-0.5', '0', '0.5', '0.99'], fontsize=12)

    ax.set_ylim(simulator.t.min(), simulator.t.max())
    ax.set_xlim(simulator.x.min(), simulator.x.max())
    ax.set_xticks([simulator.x.min(), -0.5, 0, 0.5, simulator.x.max()])
    ax.set_xticklabels(([f'{simulator.x.min()}', '-0.5', '0', '0.5', f'{simulator.x.max()}']), fontsize=14)
    ax.set_yticks([simulator.t.min(), 0.25, 0.5, 0.75, simulator.t.max()])
    ax.set_yticklabels([f'{simulator.t.min()}', '0.25', '0.5', '0.75', f'{simulator.t.max()}'], fontsize=14)
    ax.set_ylabel('$t$', size=22)
    ax.set_xlabel('$x$', size=22)

    plt.tight_layout()
    plt.draw()
    plt.show()
    # plt.savefig(f"burger_space.pdf")

    # u(t,x) at one specific t
    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)

    fig, ax = plt.subplots(sharex=True, figsize=(6, 4))
    ax.plot(simulator.x, sample[-1, :], 'ro', linewidth=2.0) #, label='Exact')
    ax.set_xlabel('$x$', fontsize=24)
    ax.set_ylabel(f'$u(x, t={int(simulator.t.max())})$', fontsize=24)
    ax.set_yticks([-2.0, -1.0, 0.0, 1.0, 2.0])
    ax.set_ylim((sample[-1,:].min() - 0.1), (sample[-1,:].max() + 0.1))
    # ax.axvline(x=simulator.x[10], color='saddlebrown', linewidth=2.5)
    # ax.axvline(x=simulator.x[-10 - 1], color='saddlebrown', linewidth=2.5)

    ax.grid(True, linewidth=0.3)

    plt.tight_layout()
    plt.draw()
    plt.show()
    # plt.savefig(f"burger_time.pdf")
    plt.close('all')

    # u(t, x) over space at a specific time unit
    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)

    fig, ax = plt.subplots()

    # Add noise to the data if you want to
    # sample[:80] = sample[:80] + np.random.normal(np.zeros_like(sample[:80]),
    #                                                 np.ones_like(sample[:80]) * 0.05)

    sample = np.transpose(sample)


    line1, = ax.plot(simulator.x, sample[:, 0], 'ro', linewidth=2, label='Exact')
    ax.set_xlabel('$x$', size=22)
    ax.set_ylabel('$u(t,x)$', size=22)
    ax.set_xlim([simulator.x.min(), simulator.x.max()])
    ax.set_yticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.grid(True, linewidth=0.3)
    cut_domain_size = 10
    ax.axvline(x=simulator.x[cut_domain_size], color='saddlebrown', linewidth=2.5)
    ax.axvline(x=simulator.x[-cut_domain_size - 1], color='saddlebrown', linewidth=2.5)
    ax.set_ylim([-1.65, 1.65])

    anim = animation.FuncAnimation(fig,
                                   animate,
                                   frames=len(simulator.t),
                                   fargs=(line1, sample),
                                   interval=20)

    plt.tight_layout()
    plt.draw()
    plt.show()

    # f = f"burger-animation.mp4"
    # writervideo = animation.FFMpegWriter(fps=60)
    # anim.save(f, writer=writervideo)

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


def main():
    """
    Main method used to create the datasets.
    """

    # Determine the root path for this script and set up a path for the data
    root_path = os.path.abspath("")
    
    # Create a wave generator using the parameters from the configuration file
    if not MINI_BATCHES:
        simulator = Simulator(
                diffusion_coefficient=DIFFUSION_COEFFICIENT,
                left_BC=LEFT_BC,
                right_BC=RIGHT_BC,
                t_max=T_MAX,
                t_steps=T_STEPS,
                x_left=X_LEFT,
                x_right=X_RIGHT,
                x_steps=X_STEPS,
                train_data=TRAIN_DATA
            )

        # Create train, validation and test data
        generate_sample(simulator=simulator,
                        visualize=VISUALIZE_DATA,
                        save_data=SAVE_DATA,
                        root_path=root_path,
                        idx=42 #it is a dummy number
                        )

    else:
        batch_size = 10
        save_left_BC = []
        save_right_BC = []
    
        for idx in range(batch_size):
            left_BC = np.random.uniform(-1,1)
            right_BC = np.random.uniform(-1,1)
            save_left_BC.append(left_BC)
            save_right_BC.append(right_BC)
            print(left_BC)
            print(right_BC)
            
            simulator = Simulator(
                diffusion_coefficient=DIFFUSION_COEFFICIENT,
                left_BC=left_BC,
                right_BC=right_BC,
                t_max=T_MAX,
                t_steps=T_STEPS,
                x_left=X_LEFT,
                x_right=X_RIGHT,
                x_steps=X_STEPS,
                train_data=TRAIN_DATA
            )
    
            # Create train, validation and test data
            generate_sample(simulator=simulator,
                            visualize=VISUALIZE_DATA,
                            save_data=SAVE_DATA,
                            root_path=root_path,
                            idx=idx
                            )
        name = "left_BC.npy"
        data_path = os.path.join(root_path, DATASET_NAME+"_train")
        np.save(file=os.path.join(data_path, name), arr=save_left_BC)
        
        name = "right_BC.npy"
        data_path = os.path.join(root_path, DATASET_NAME+"_train")
        np.save(file=os.path.join(data_path, name), arr=save_right_BC)

        print(f'Left BCs:  {save_left_BC}')
        print(f'Right BCs: {save_right_BC}')

if __name__ == "__main__":
    main()

    print("Done.")
