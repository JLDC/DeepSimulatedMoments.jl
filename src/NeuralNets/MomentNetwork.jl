export MomentNetwork, apply_transforms, generate, train_network!

"""
    MomentNetwork

A neural network that estimates the parameters of a DGP.

# Fields
- `model`: The neural network.
- `optimizer`: The optimizer.
- `best_model`: The best model found during training.
- `hyperparameters`: The hyperparameters.
- `preprocess`: The preprocessing function.
- `parameter_transform`: The parameter transform.
- `data_transform`: The data transform.
"""
mutable struct MomentNetwork
    model
    optimizer
    best_model
    hyperparameters::HyperParameters
    preprocess::Function
    parameter_transform::Union{Nothing,AbstractDataTransform}
    data_transform::Function
end

"""
    MomentNetwork(model, optimizer, hyperparameters; 
        preprocess, parameter_transform, data_transform)

Create a [`MomentNetwork`](@ref).

# Arguments
- `model`: The neural network.
- `optimizer`: The optimizer.
- `hyperparameters`: The hyperparameters.

# Keyword Arguments
- `preprocess`: The preprocessing function (default: `nothing`).
- `parameter_transform`: The parameter transform (default: `nothing`).
- `data_transform`: The data transform (default: `tabular2conv`).
"""
function MomentNetwork(
    model, optimizer, hyperparameters::HyperParameters;
    preprocess=nothing, parameter_transform=nothing, data_transform=tabular2conv)
    preprocess = isnothing(preprocess) ? (x, y) -> (x, y) : preprocess
    MomentNetwork(model, optimizer, deepcopy(model), hyperparameters, preprocess, parameter_transform, data_transform) # TODO: datatransform
end

has_parameter_transform(net::MomentNetwork) = !isnothing(net.parameter_transform)

"""
    apply_transforms(net::MomentNetwork, X, Y)

Apply the data and parameter transforms to the data and parameters, respectively.

# Arguments
- `net::MomentNetwork`: The moment network.
- `X`: The data to apply the data transform to.
- `Y`: The parameters to apply the parameter transform to.

# Returns
- `X`: The transformed data.
- `Y`: The transformed parameters.
"""
function apply_transforms(net::MomentNetwork, X, Y)
    X = X |> net.data_transform
    has_parameter_transform(net) && StatsBase.transform!(net.parameter_transform, Y)
    net.preprocess(X, Y)
end


"""
    generate(dgp::AbstractDGP, net::MomentNetwork, nsamples::Int)

Generate data from a DGP and apply the data and parameter transforms from the moment network.

# Arguments
- `dgp::AbstractDGP`: The DGP to generate data from.
- `net::MomentNetwork`: The moment network.
- `nsamples::Int`: The number of samples to generate.

# Returns
- `X`: The transformed data.
- `Y`: The transformed parameters.
"""
function generate(dgp::AbstractDGP, net::MomentNetwork, nsamples::Int)
    X, Y = generate(dgp, nsamples, dev=net.hyperparameters.dev)
    apply_transforms(net, X, Y)
end

_nlosses(net::MomentNetwork) = div(net.hyperparameters.epochs * net.hyperparameters.nsamples,
    net.hyperparameters.validation_freq)

"""
    train_network!(net::MomentNetwork, dgp::AbstractDGP; verbose::Bool=true)

Train a moment network on a DGP.

# Arguments
- `net::MomentNetwork`: The moment network.
- `dgp::AbstractDGP`: The DGP to train on.
- `verbose::Bool`: Whether to print information about the training process.

# Returns
- `iterations`: The iterations at which the losses were computed.
- `losses`: The losses at each iteration.
"""
function train_network!(
    net::MomentNetwork, dgp::AbstractDGP{T}; 
    verbose::Bool=true
) where {T<:AbstractFloat}
    Flux.trainmode!(net.model) # Ensure that the model is in training mode
    θ = Flux.params(net.model) # Extract model parameters
    
    net.best_model = deepcopy(net.model) # Reset the best model
    best_loss = Inf # Initialize the best loss

    validation = net.hyperparameters.validation_size > 0
    losses = zeros(T, _nlosses(net))
    iterations = zeros(Int, _nlosses(net))
    loss_iteration = 1

    # Handle validation
    if validation
        Xval, Yval = generate(dgp, net, net.hyperparameters.validation_size)
        
        verbose && @info "Validation set size: $(size(Yval, 2))"
    end

    if verbose # Compute pre-training loss
        Ŷ = net.model(Xval)
        loss = net.hyperparameters.loss(Ŷ, Yval)
        @info "Pre-training loss: $(loss)"
    end

    for epoch ∈ 1:net.hyperparameters.epochs
        for sample ∈ 1:net.hyperparameters.nsamples
            iteration = (epoch - 1) * net.hyperparameters.nsamples + sample
            # Generate training data
            X, Y = generate(dgp, net, net.hyperparameters.batchsize)

            # Gradient step
            ∇ = gradient(θ) do 
                Ŷ = net.model(X)
                net.hyperparameters.loss(Ŷ, Y)
            end
            Flux.update!(net.optimizer, θ, ∇)

            # Handle validation
            if validation && mod(iteration, net.hyperparameters.validation_freq) == 0
                Ŷ = net.model(Xval)
                
                loss = net.hyperparameters.loss(Ŷ, Yval)
                losses[loss_iteration] = loss
                iterations[loss_iteration] = iteration
                loss_iteration += 1

                if loss < best_loss
                    best_loss = loss
                    net.best_model = deepcopy(net.model)
                end

                if verbose && mod(iteration, net.hyperparameters.print_every) == 0
                    @info "Sample: $(iteration), loss: $(loss)"
                end

            else
                if verbose && mod(iteration, net.hyperparameters.print_every) == 0
                    @info "Sample: $(iteration)"
                end
            end
        end
    end
    iterations, losses
end