# Koopman operator learning for Tropical Cylones

In this repository I test the applicability of the Koopman operator framework to analyse
and predict the dynamics of tropical cylones. Based on these prediction models, I try to
assess risk and impact of future cyclone events on certain economic exposures.

The Koopman operator allows to capture highly complex, nonlinear dyanmics in a infinite-dimensional linear operator.
Based on this representation, we can for example perform spectral analysis of nonlinear systems. 

For this study, the [CLIMADA](https://github.com/CLIMADA-project/climada_python) python package serves as a perfect starting point, since it
provides a huge amount of tropical cylone track data, and has a lot of build in functionalities
for risk assessment and economic exposure analysis. The training of the Koopman model is 
based on the [kooplearn](https://github.com/Machine-Learning-Dynamical-Systems/kooplearn) python package.

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

