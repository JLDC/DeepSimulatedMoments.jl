export HyperParameters

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

function HyperParameters(;
    batchsize::Int=32, epochs::Int=1, dev::Function=cpu, 
    loss::Function=Flux.Losses.mse, validation_size::Int=10_000, validation_freq::Int=10,
    print_every::Int=100, nsamples::Int=100
)
    HyperParameters(dev, loss, batchsize, nsamples, epochs, validation_size, 
        validation_freq, print_every)
end