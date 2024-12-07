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

To install torch CUDA, please follow the instructions at https://pytorch.org/get-started/locally/ depending on your CUDA version.s

For Compute Platform CUDA 12.4 and higher one can use the following

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```



## Project description

### Tropical cylones
We use the CLIMADA package to import tropical cyclone dataset and for plotting. There are five main basins of tropical cylones which we consider here: East Pacific (EP), North Atlantic (NA), South Indian Ocean (SI), South Pacific (SP) and West Pacific (WP).

Below we show example tracks of tropical cyclones in the East Pacific (left) and the North Atlantic (right).
![Tropical cyclone track](./plots/tropical_cyclone_data/TCTrack_EP_NA.png)
The color of the track indicate the severity of the cyclone/hurricane, which is classified into seven severity levels from "Tropical Depression" to "Hurrican Category 5", see legend.

In the following we train separate Koopman Kernel models on each of the basins. Based on these models we can predict the future tracks and severity of new tropical cyclones, which can be used for downstream tasks such as risk assessment and economical impact calculations for such events.
Another aspect is the dynamical characterization of the different basins via a spectral analysis of the corresponding Koopman operators. The Koopman modes characterize the dynamics and can be used to idenfity and discriminate dynamical signatures of the basins.



### Spectral consistency analysis and implied time scale:
From the spectrum of the Koopman operator one can obtain the time scales of the dynamical modes of the system. An important test for spectral consistency considers the scaling of implied time scales (ITS) as we vary the internal time unit of the dynamics. This can be controlled by adding a `time_lag` between the observation points.
As we increase the internal time unit of the system, the eigenvalues should become smaller and the associated dyanmics should become faster (i.e. a decrease of the time scale). However, the implied time scale, which is the product of time lag and time scale, should stay relatively constant.

![Timelag scaling of ITS](./plots/koopman_spectral_analysis/time_lag_scaling/time_lag_scaling_ctlen4.png)

In the left plot we observe that the RMSE monotonically increases with the time lag, but with decreasing slope.
For large time lags it might eventually converge to a fixed value.
In the right plot we show the implied time scales from five selected eigenvalues of the Koopman operator.
We observe a typical behavior, where initially the ITS increases, but for larger time lags the ITS converges to a constant value. This verifies the consistency of our model. As we increase the implicit time unit of our model (controlled by the time lag of the training data) the ITS should stay constant.
Additionally we observe a strong fluctuation of the largest shown eigenvalue (smallest index), which is very close to the theoretical maximum of one.
This shows that our eigenvalue computation is unstable close to the steady state (corresponding to the eigenvalue one), which is, however, expected when training the Koopman kernel models on limited real world data. The lower eigenvalues (larger index) show a much more stable behavior.


### Plan:

In this context, I aim to test two different Koopman-based approaches. The first ansatz follows a kernel-based approach for
approximating Koopman operators in reproducing kernel Hilberg spaces (see [kooplearn](https://github.com/Machine-Learning-Dynamical-Systems/kooplearn) python package).
Secondly, I want to investigate the recent [Koopman Neural Forecaster](https://github.com/google-research/google-research/tree/master/KNF) architecture.
This architecture is build from a local and a global Koopman operator (implemented as trainable deep neural networks (multi-layer perceptron and
transformers)), both capturing local respectively global behaviour of the time series. This is combined with a feedback loop,
that is designed to capture and correct for spontaneous, sudden shifts and distortions in the temporal distribution.

For the Koopman Neural Forecaster, I plan to test several potential improvement directions. For example the selection of observable functions seems sub-optimal. There is some redundancy in the observable functions and there are only very few non-linear functions (discussed in more detail below), which are crucial for learning non-linear dynamics. Apart from reducing redundancy and increasing the non-linearity content of the observable functions, I aim to replace the (slow) attention mechanism with random feature kernels as described in [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794).

### Interesting questions:

Observables/measurement functions: Almost all observables of the KNF are linear functions of the system state $x$ and do not couple the coordinates $x_i$, for example of the
form $\sin(\sum_i c_i x_j(t_i))$ with learned coefficients $c_i$. The $j$-th coordinate $x_j(t_i)$ of the dynamical state is evaluated in a short time-window $[t_0, t_1, \dots]$
discretized by the $t_i$. Hence, the argument of the $\sin$ does not couple different coordinates, but it is only a linear superposition of the same coordinate evaluated at some previous time steps.
Furthermore there is a substantial redundancy in the measurement functions. For example, the function $\sin(\sum_i c_i x_j(t_i))$ is included multiple times in the set of observables, each time with separate trainable $c_i$.
The only non-linear functions included in the observables are simple products (only second order) of the form $x_i x_j$.

It would be interesting to improve the set of observables, i.e. to reduce the redundancy (-> improve time complexity by keeping same performance) and to include more (and maybe more suitable)
nonlinear observables, i.e. interactions (-> improve performance). So far the interactions $x_i x_j$ are the only observables that couple the coordinates.


### Contributions

#### Model improvements
- Implementation of additive combination of global and local Koopman operator:
    - although the authors described this additive combination in the paper, they implemented multiplicative combination in the code
    - additive combination seems much more natural from a perspective based on Koopman operator theory
    - in first tests, additive combination performs better than the multiplicative option
- Comparison between Koopman kernel regression and deep neural network transformer architecture

#### Data processing
- Conversion of CLIMADA TCTracks into kooplearn and pytorch compatible data structures
- Data quality and extract characteristic length scale from data (needed for kooplearn kernal models)
- Data standardization and periodic centering of earth scale tropical cyclone data (discontinuous cut along the longitudinal coordinate for $\mathrm{lon} = +- 180^\circ$)

