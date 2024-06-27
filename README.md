# entanglement_partitions_paper_freiburg
Code for data generation for the entanglement partition paper (with Aaron and Gabriel)

## Setup
To install the project, first clone the repository and then run from the project folder:
```
python -m venv .venv
```
To activate the environment, run
```
source .venv/bin/activate
```
on Mac and Linux, or
```
.venv\Scripts\activate
```
on Windows.

Then, run
```
pip install -e .
```
to install the project.


## Data

Download the provided data and copy the folders `obs`, `ppVs`, `states` and `Vs` into the folder `data/mathematica/`. The folder `obs` contains a few on-site densities and two-point correlators, these can be easily generated with the code used e.g. to build the Hamiltonian. The folder `states` contains states of partially distinguishable particles with random internal states of the particle, i.e. for varying partial distinguishability. For `N` particles on `L>=N` sites, the initial state in the external modes is always `a^\dagger_1 \dots a^\dagger_N \ket{0}`, i.e. a single particle per external mode. The folders `Vs` and `ppVs` contain projectors onto the Young symmetry sectors. These can be also generated again with the provided code.

## Folders

The folder `examples/notebooks` contains some notebooks showing how to generate the projectors onto the Young symmetry sectors and how to define a extended Bose Hubbard Hamiltonian.

The folder `examples/scripts` contains a few scripts generating data, such as time evolution, temporal variance, purities and other potentially relevant quantities. Run the files ending with `..._test.py` to see if everyting is working. A folder with data should be generated within `data/`.

The other files contain larger simulations with corresponding `.slurm` which probably need to be executed on a cluster. Don't forget to change the email address field in the `.slurm` files.
