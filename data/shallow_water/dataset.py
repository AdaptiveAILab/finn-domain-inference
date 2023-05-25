"""
This file contains the organization of the custom datasets such that it can be
read efficiently in combination with the DataLoader from PyTorch to prevent that
data reading and preparing becomes the bottleneck.

This script was inspired by
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
"""

from data.datasets.dataset import Dataset

__author__ = "Manuel Traub"


class ShallowWaterDataset(Dataset):
    """
    The custom wave datasets class which can be used with PyTorch's DataLoader
    """

    def __init__(self, root_path, dataset_name, dataset_type):
        """
        Constructor class setting up the data loader
        :param root_path: The root path of the project
        :param dataset_name: The name of the datasets (e.g. "simple_wave")
        :param dataset_type: Any of "train", "val" or "test"
        :return: No return value
        """
        super(ShallowWaterDataset, self).__init__("shallow_water", root_path, dataset_name, dataset_type)

    def getitem(self, data: dict):
        """
        Generates a sample batch in the form [time, batch_size, x, y, dim],
        where x and y are the sizes of the data and dim is the number of
        features.
        :param data: Dictionary with data
        :return: One batch of data as np.array
        """

        # Load a sample from file and divide it in input and label. The label
        # is the input shifted one timestep to train one step ahead prediction.
        sample = data['labels']
        net_input = sample[:-1]
        net_label = sample[1:]

        net_input = net_input.reshape((net_input.shape[0], 1, net_input.shape[-2], net_input.shape[-1]))
        net_label = net_label.reshape((net_input.shape[0], 1, net_label.shape[-2], net_label.shape[-1]))
        return net_input, net_label
