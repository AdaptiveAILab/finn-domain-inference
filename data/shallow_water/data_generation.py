"""
This script creates a certain amount of 2D wave data. The wave data can be
modified such that particular cells have an activity absorbing or generating
effect. Parameters for the wave generation are stored in a separate config.ini
file.
"""

import torch as th
import numpy as np
import argparse
import sys
from matplotlib import animation, pyplot as plt
import os

from configuration import Configuration
from utils import Hdf5File
from plot import show3d
from simulator import Simulator

__author__ = "Manuel Traub"


def generate_dataset(simulator, dataset: Hdf5File, config, data_type, n_samples):
    """
    This function generates a datasets of a specified type (train, val or test),
    visualizes it if desired and saves the data to file if desired.
    :param simulator: The simulator.py object for data creation
    :param dataset: The dataset where the data will be saved
    :param config: The configuration object
    :param data_type: Any of "train", "val" or "test"
    :param n_samples: The number of samples for the specified data type
    :return: No return value
    """

    samples = list()

    # Create the desired number of samples
    for sample_number in range(n_samples):

        # Print a progress statement to the console for every 20 generated files
        if (sample_number + 1) % 5 == 0:
            print(f"Generating {data_type} file {sample_number + 1}/{n_samples}")

        # Generate the actual sample
        sample = simulator.generate_sample()

        # If specified, visualize the sample
        if config.dataset.visualize:
            _sample = th.tensor(sample).unsqueeze(1).unsqueeze(0)
            show3d(_sample) #visualize_sample(sample) #

        samples.append(sample)

    # If specified, write the sample to file
    if config.dataset.save_data:
        dataset.write_dataset(name=data_type, data={'labels': np.stack(samples)})


def visualize_sample(sample):
    """
    Method to visualize a single sample.
    :param sample: The actual data sample for visualization
    :return: No return value
    """

    timesteps = len(sample)

    # plt.style.use("dark_background")
    # Plot the wave activity at one position
    fig, ax = plt.subplots(1, 1, figsize=[8, 2])
    ax.plot(range(len(sample)), sample[:, 12, 12])
    ax.set_xlabel("Time")
    ax.set_ylabel("Wave amplitude")
    ax.set_xlim([0, timesteps - 1])
    plt.tight_layout()
    plt.show()

    # Animate the spatiotemporal wave
    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    im = ax.imshow(sample[0, :, :], vmin=-0.6, vmax=0.6, cmap="Blues")
    anim = animation.FuncAnimation(fig,
                                   animate,
                                   frames=timesteps,
                                   fargs=(im, sample),
                                   interval=20)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def animate(t, im, field):
    """
    Data animation function animating an image over time.
    :param t: The current time step
    :param im: The matplotlib image object
    :param field: The data field
    :return: The matplotlib image object updated with the current time step's
        image date
    """
    im.set_array(field[t, :, :])
    return im


def main():
    """
    Main method used to create the datasets.
    :return: No return value
    """
    
    # Read the configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", default="cfg.json")
    args = parser.parse_args(sys.argv[1:])
    config = Configuration(args.cfg)

    dataset = Hdf5File(path="./", filename=config.dataset.name)

    # Create a wave generator using the parameters from the configuration file
    simulator = Simulator(
        timesteps=config.simulation.timesteps,
        width=config.simulation.width,
        height=config.simulation.height,
        sample_interval=config.simulation.sample_interval,
        area_scaling=config.simulation.area_scaling,
        quantity_scaling=config.simulation.quantity_scaling
    )
    
    
    # Create train, validation and test data
    generate_dataset(
        simulator=simulator,
        dataset=dataset,
        config=config,
        data_type="train",
        n_samples=config.dataset.samples_train)
    
    generate_dataset(
        simulator=simulator,
        dataset=dataset,
        config=config,
        data_type="val",
        n_samples=config.dataset.samples_val)
     
    generate_dataset(
        simulator=simulator,
        dataset=dataset,
        config=config,
        data_type="test",
        n_samples=config.dataset.samples_test)

    dataset.close()
    
    
if __name__ == "__main__":
    main()

    print("Done.")
