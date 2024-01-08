using DeepSimulatedMoments
using Flux
using StatsBase
using BSON

# choose the DGP
dgp = MA2(N=100)

# configure the TCN
tcn = build_tcn(dgp, kernel_size=32, channels=64); # Build a TCN for this DGP

# define the loss function
rmse_conv(Ŷ, Y) = mean(sqrt.(mean(abs2.(Ŷ - Y), dims=2)))

# function to create the net
function make_net(dev)
    # Set up the hyperparameters
    hp = HyperParameters(
        validation_size=2_000, loss=rmse_conv, 
        print_every=10, nsamples=100, epochs=1, dev=dev, batchsize=128
    )
    # Create the net
    net = MomentNetwork(
        tcn |> hp.dev, ADAMW(), hp
    )
end

# Create and train the net, and save final state
function train_net(dgp)
    net = make_net(cpu)
    train_network!(net, dgp)
    net_state = Flux.state(net)
    BSON.@save "net_state.bson"  net_state
end

# Recreate trained net on CPU to do Bayesian MSM
function load_trained()
    isfile("net_state.bson") ? @info("loading trained net") : @info("load_trained: can't create net on CPU: net_state has not been saved")
    net = make_net(cpu) # recreate the net
    BSON.@load "net_state.bson"  net_state # get the trained parameters
    Flux.loadmodel!(net, net_state)
    Flux.testmode!(net)
    return net
end
    

# train_net(dgp)  # comment out when net is trained


net = load_trained()

data, θtrue = generate(dgp, net, 1)
θnn = net.best_model(data)[:]

# set up proposal
covreps = 1000
_,Σₚ = simmomentscov(net, dgp, covreps, θnn)
δ = 1.0 # tuning

# do MCMC
S = 50 
chain = mcmc(θnn, θnn, δ, Σₚ, S, net, dgp) 
