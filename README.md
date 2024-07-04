# Koopman operator learning for Tropical Cylones

In this repository I test the applicability of the Koopman operator framework to predict
and characterise dynamics of tropical cylones. Based on these prediction models, we can 
project the economic impact of future cylones.

To access data and the climate economic impact analysis I rely on the CLIMADA python package.
For training Koopman operator models, I use the Kooplearn package.

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

