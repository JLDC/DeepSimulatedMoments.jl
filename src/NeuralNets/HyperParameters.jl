export HyperParameters

"""

    HyperParameters

Hyperparameters for training a [`MomentNetwork`](@ref).

# Fields
- `dev`: The device to train on.
- `loss`: The loss function.
- `batchsize`: The batch size.
- `nsamples`: The number of samples to generate.
- `epochs`: The number of epochs to train for.
- `validation_size`: The number of samples to use for validation.
- `validation_freq`: The frequency with which to validate.
- `print_every`: The frequency with which to print information about the training process.
"""
mutable struct HyperParameters
    dev::Function
    loss::Function
    batchsize::Int
    nsamples::Int
    epochs::Int
    validation_size::Int
    validation_freq::Int
    print_every::Int
end

"""

    HyperParameters(; batchsize::Int=32, epochs::Int=1, dev::Function=cpu, 
        loss::Function=Flux.Losses.mse, validation_size::Int=10_000, validation_freq::Int=10,
        print_every::Int=100, nsamples::Int=100)

Construct a [`HyperParameters`](@ref) structure.

# Arguments
- `batchsize::Int`: The batch size (default=`32`).
- `epochs::Int`: The number of epochs to train for (default=`1`).
- `dev::Function`: The device to train on (default=`cpu`).
- `loss::Function`: The loss function (default=`Flux.Losses.mse`).
- `validation_size::Int`: The number of samples to use for validation (default=`10_000`).
- `validation_freq::Int`: The frequency with which to validate (default=`10`).
- `print_every::Int`: The frequency with which to print information about the training process (default=`100`).
- `nsamples::Int`: The number of samples to generate (default=`100`).
"""
function HyperParameters(;
    batchsize::Int=32, epochs::Int=1, dev::Function=cpu, 
    loss::Function=Flux.Losses.mse, validation_size::Int=10_000, validation_freq::Int=10,
    print_every::Int=100, nsamples::Int=100
)
    HyperParameters(dev, loss, batchsize, nsamples, epochs, validation_size, 
        validation_freq, print_every)
end