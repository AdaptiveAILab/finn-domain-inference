import h5py
import os
import numpy as np

from typing import Dict


class Hdf5File:
    """
    IO for making a hdf5 dataset
    """
    def __init__(self, path: str, filename: str, overwrite=True):
        os.makedirs(path, exist_ok=True)
        self.path = os.path.join(path, f'{filename}.hdf5')

        self.dataset = h5py.File(self.path, "w" if overwrite else "w-")

    def close(self):
        """
        Closes the dataset
        """
        self.dataset.flush()
        self.dataset.close()

        self.dataset = None

    def write_dataset(self, name: str, data: Dict[str, np.ndarray]):
        """
        Makes a group and adds datasets
        :param name: Name of the group
        :param data: Datasets with name and data to be added
        """
        group = self.dataset.create_group(name)

        for name in data:
            shape = list(data[name].shape)
            shape[0] = 1
            group.create_dataset(name=name, data=data[name], compression='gzip', chunks=tuple(shape))

        self.dataset.flush()
