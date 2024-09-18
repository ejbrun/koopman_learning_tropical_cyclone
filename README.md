# Koopman operator learning for Tropical Cylones

[Work in progress]

In this repository I test the applicability of the Koopman operator framework to analyse
and predict the dynamics of tropical cylones. Based on these prediction models, I plan to 
assess the economical impact and risk associated with future cyclone events.

The Koopman operator allows to capture highly complex, nonlinear dyanmics in a infinite-dimensional linear operator.
Based on this representation, we can for example perform spectral analysis of nonlinear systems. 

For this study, the [CLIMADA](https://github.com/CLIMADA-project/climada_python) python package serves as a perfect starting point, since it
provides a huge amount of tropical cylone track data, and has a lot of build in functionalities for risk assessment and economic exposure analysis.

The aim is to test two different Koopman-based approaches. The first approach follows the kernel methods implemented in the
[kooplearn](https://github.com/Machine-Learning-Dynamical-Systems/kooplearn) python package. Secondly, I want to investigate
the recent [Koopman Neural Forecaster](https://github.com/google-research/google-research/tree/master/KNF). This architecture
is build from a local and a global Koopman operator (implemented as trainable deep neural networks), both capturing local respectively
global behaviour of the time series. This is combined with a feedback loop, that is designed to capture and correct for spontaneous,
sudden shifts and distortions in the temporal distribution. The local Koopman operator is basically 


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

