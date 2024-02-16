# Moment Generating Networks

Moment Networks are the neural networks that create the moment conditions for the method of simulated moments (MSM). The networks learn a mapping from the data to the parameters that generated this data. 

## Creating a Moment Network


Moment networks are created using the [`MomentNetwork`](@ref) constructor. The constructor takes the following arguments:

+ `model`: The neural network, e.g., a [`TCN`](@ref). More generally, any kind of neural network built using [Flux.jl](https://fluxml.ai/Flux.jl/dev/)
+ `optimizer`: The optimizer, e.g., `Flux.Optimise.ADAM`. More generally, any kind of optimizer built using [Optimisers.jl](https://fluxml.ai/Optimisers.jl/dev/)
+ `hyperparameters`: The hyperparameters, a special structure of this package, [`HyperParameters`](@ref), which governs the training hyperparameters of the moment network.
+ `preprocess`: The preprocessing function. This function is applied to the data and parameters before they are passed to the neural network. The default is `nothing`, which means that no preprocessing is applied.
+ `parameter_transform`: The parameter transform. This transform is applied to the parameters before they are passed to the neural network. The default is `nothing`, which means that no transform is applied. Note that if the ranges of the parameters vary greatly, it is recommended to use a parameter transform, e.g., [`datatransform`](@ref).
+ `data_transform`: The data transform. This transform is applied to the data before they are passed to the neural network. The default is `tabular2conv`, which means that the data is transformed from a tabular format to a convolutional format, which can be readily used by a [`TCN`](@ref).

### Hyperparameters

The training hyperparameters are stored in a [`HyperParameters`](@ref) structure. This structure contains the following fields:
+ `dev`: The device on which the moment network is trained. The default is `cpu`, which means that the moment network is trained on the CPU. Alternatively, if a GPU is available, `gpu` can be used.
+ `loss`: The loss function. The default is the mean-square-error.
+ `batchsize`: The batch size. For each sample (see below), a batch of size `batchsize` is drawn.
+ `nsamples`: The number of samples generated per epoch.
+ `epochs`: The number of epochs.
+ `validation_size`: The size of the validation set. 
+ `validation_freq`: The frequency with which the validation set is evaluated.
+ `print_every`: The frequency with which information about the training process is printed.

## Training the Network

The moment network is trained using the [`train_network!`](@ref) function. This function takes the following arguments:  
+ `net`: The moment network to be trained.
+ `dgp`: The data generating process. This is an object of type [`AbstractDGP`](@ref).
+ `verbose`: A boolean indicating whether the training process should be verbose. The default is `true`.

During training, validation occurs exclusively at intervals defined by the `validation_freq` parameter defined by the [`HyperParameters`](@ref) structure. At each specified interval, the framework performs validation, records the loss, and stores these values. The function outputs two arrays upon completion:   
+ `iterations`: An array of the iterations at which validation occurred.
+ `losses`: An array of the corresponding losses recorded at those iterations.

### Example
The following code snippet shows how to create and train a moment network for an [`MA2`](@ref) data generating process:

```julia
using Flux # For ADAMW(), the optimizer
dgp = MA2(100) # Create an MA(2) DGP with 100 observations
tcn = build_tcn(dgp) # Build a TCN for this DGP

# Set up the hyperparameters
hp = HyperParameters(
    validation_size=1_000, loss=rmse_conv, 
    print_every=5, nsamples=100, epochs=5,
)

# Create the moment network
net = MomentNetwork(
    tcn |> hp.dev, ADAMW(), hp, 
    parameter_transform=(datatransform(dgp, 100_000, dev=hp.dev))
)

# Train the moment network
iterations, losses = train_network!(net, dgp)
```