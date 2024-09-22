# Koopman operator learning for Tropical Cylones

[Work in progress]

In this repository I test the applicability of the Koopman operator framework to analyse
and predict the dynamics of tropical cylones. Based on these prediction models, we can 
assess the economical impact and risk associated with future cyclone events.

The Koopman operator allows to capture complex, nonlinear dyanmics via a infinite-dimensional linear operator.
Finite dimensional approximations of the Koopman operator can be used for example spectral analysis of nonlinear
systems and forecasting of nonlinear dynamics.

For this study, the [CLIMADA](https://github.com/CLIMADA-project/climada_python) python package serves as a starting point. It
provides tropical cylone track data, and has build-in functionalities for risk assessment and economic exposure analysis.
In this context, I aim to test two different Koopman-based approaches. The first ansatz follows a kernel-based approach for
approximating Koopman operators in reproducing kernel Hilberg spaces (see [kooplearn](https://github.com/Machine-Learning-Dynamical-Systems/kooplearn) python package).
Secondly, I want to investigate the recent [Koopman Neural Forecaster](https://github.com/google-research/google-research/tree/master/KNF) architecture.
This architecture is build from a local and a global Koopman operator (implemented as trainable deep neural networks (multi-layer perceptron and
transformers)), both capturing local respectively global behaviour of the time series. This is combined with a feedback loop,
that is designed to capture and correct for spontaneous, sudden shifts and distortions in the temporal distribution.


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

