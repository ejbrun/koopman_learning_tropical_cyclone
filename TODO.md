
### TODOs

- [x] conversion of tropical cyclone data into pytorch dataloader framework
- [x] test KNF on tropical cyclone data
- [x] visualize predicted curves from klearn and KNF
- [ ] (low priority) implement model prediction for time series of arbitrary length (at the moment a certain context length of the dataset is fixed, koopman kernel model is only able to predict for time series of that length)

Model improvements:
- [ ] exchange the attention mechanism with a more efficient kernel attention mechanism
- [ ] consider observables, i.e. more non-linear observables, less redundancy in the set of observables
- [x] local Koopman operator: Currently, it seems like the local Koopman operator (modelled as the weight matrix of a transformer encoder layer) is applied multiplicatively after the global Koopman operator, not additive. According to the paper it should be additive, test this.
