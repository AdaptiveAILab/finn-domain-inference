# LabRotation-Horuz
This is the repository for Coşku's lab rotation about inferring unknown boundary conditions when modeling spatio-temporal PDEs with FINN.

This repository contains the PyTorch code for models, training, testing, and Python code for data generation to conduct the experiments as reported in the work [Inferring Boundary Conditions in Finite Volume Neural Networks](https://link.springer.com/chapter/10.1007/978-3-031-15919-0_45)

### Abstract

When modeling physical processes in spatially confined domains, the boundaries require distinct consideration through specifying appropriate boundary conditions (BCs). The finite volume neural network (FINN) is an exception among recent physics-aware neural network models: it allows the specification of arbitrary BCs. FINN is even able to generalize to modified BCs not seen during training, but requires them to be known during prediction. However, so far even FINN was not able to handle unknown BC values. Here, we extend FINN in order to infer BC values on-the-fly. This allows us to apply FINN in situations, where the BC values, such as the inflow rate of fluid into a simulated medium, is unknown. Experiments validate FINN’s ability to not only infer the correct values, but also to model the approximated Burgers’ and Allen-Cahn equations with higher accuracy compared to competitive pure ML and physics-aware ML models. Moreover, FINN generalizes well beyond the BC value range encountered during training, even when trained on only one fixed set of BC values. Our findings emphasize FINN’s ability
to reveal unknown relationships from data, thus offering itself as a process-explaining system.

`Keywords:` Physics-aware neural networks · Boundary conditions · Retrospective inference · Partial differential equations · Inductive biases.

If you find this repository helpful, please cite our work:

```
@InProceedings{Horuz2022,
    author    = {Coşku Can Horuz and Matthias Karlbauer and Timothy Praditia and Martin V. Butz and Sergey Oladyshkin and Wolfgang Nowak and Sebastian Otte},
    booktitle = {International Conference on Artificial Neural Networks (ICANN)},
    title     = {Inferring Boundary Conditions in Finite Volume Neural Networks},
    year      = {2022},
    publisher = {Springer Nature Switzerland},
    isbn      = {978-3-031-15919-0},
    pages     = {538--549},
    groups    = {confpeer}
}
```

### Dependencies

We recommend setting up an (e.g. [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)) environment with python 3.7 (i.e. `conda create -n finn python=3.7`). The required packages for data generation and model evaluation are

  - `conda install -c anaconda numpy scipy`
  - `conda install -c pytorch pytorch==1.9.0`
  - `conda install -c jmcmurray json`
  - `conda install -c conda-forge matplotlib torchdiffeq jsmin`

### Models & Experiments

The code of the pure machine learning model DISTANA and physics-aware models (PhyDNet and FINN) can be found in the `models` directory.

Each model directory contains a `config.json` file to specify model parameters, data, etc. Please modify the sections in the respective `config.json` files.


The actual models can be trained and tested by calling the according `python train.py` or `python test.py` scripts. Alternatively, `python experiment.py` can be used to either train or test n models (please consider the settings in the `experiment.py` script).

### Data generation

The Python scripts to generate the burger and allen-cahn data can be found in the `data` directory. In each of these directories, a `data_generation.py` and `simulator.py` script can be found. The former is used to generate train, extrapolation (ext), or test data. For details about the according data generation settings of each dataset, please refer to the corresponding data sections in the paper.
