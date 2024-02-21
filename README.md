# DeepSimulatedMoments

| **Documentation** | **Build Status** |
|:-----------------:|:----------------:|
|[![Docs][docs-img]][docs-url]|[![Build Status][status-img]][status-url][![Codecov][codecov-img]][codecov-url]|



[codecov-img]: https://codecov.io/gh/JLDC/DeepSimulatedMoments.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JLDC/DeepSimulatedMoments.jl

[status-img]: https://github.com/JLDC/DeepSimulatedMoments.jl/actions/workflows/CI.yml/badge.svg?branch=main
[status-url]: https://github.com/JLDC/DeepSimulatedMoments.jl/actions/workflows/CI.yml?query=branch%3Amain

[docs-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-url]: https://jldc.github.io/DeepSimulatedMoments.jl


DeepSimulatedMoments.jl provides an implementation of the methods proposed in [*Constructing Efficient Simulated Moments Using Temporal Convolutional Networks* by Chassot, J. and Creel, M. (2023)](https://www.jldc.ch/uploads/2023_chassot_creel.pdf). The package allows to define your own data-generating processes, set up and train neural networks to generate moment conditions and proceed with simulation-based inference. For information on how to use this package, please refer to the [documentation](docs-url)

## Quick Example
```julia
using DeepSimulatedMoments
using Flux # Flux provides the optimizer used in this example, ADAMW

# Create a moving-average process of order 2 with n=100 observations
dgp = MA2(100)
# Build a TCN to generate moment conditions for this DGP
tcn = build_tcn(dgp)

# Set up hyperparameters
hp = HyperParameters(
    validation_size=1_000, # Use 1'000 samples to validate the final network
    loss=rmse_conv, # The loss function to use in the training of the network
    nsamples=100, # Number of samples (of `n=100` observations) per epoch
    epochs=5, # Number of total epochs
    print_every=5, # Print train/test loss every 5 samples
    dev=cpu # Use the CPU as a device for the network
)

# Create the moment network
net = MomentNetwork(
    tcn |> hp.dev, # Specify the network to use and pass the TCN to the device
    hp, # Specify the hyperparameters used for training and validation
    ADAMW(), # Specify the optimizer used for training
    # Specify a transformation applied to the parameters of the DGP pre-training
    parameter_transform=datatransform(dgp, 100_000, dev=hp.dev)
)
```