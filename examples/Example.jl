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
        validation_size=5_000, loss=rmse_conv, 
        print_every=10, nsamples=10000, epochs=2, dev=dev, batchsize=256
    )
    # Create the net
    net = MomentNetwork(
        tcn |> hp.dev, ADAMW(), hp
    )
end

# Create and train the net, and save final state
function train_net(dgp)
    net = make_net(gpu)
    train_network!(net, dgp)
    net_state = cpu(Flux.state(net.best_model))
    BSON.@save "net_state.bson"  net_state
end

# Recreate trained net on CPU to do Bayesian MSM
function load_trained()
    isfile("net_state.bson") ? @info("loading trained net") : @info("load_trained: can't create net on CPU: net_state has not been saved")
    net = make_net(cpu) # recreate the net
    BSON.@load "net_state.bson"  net_state # get the trained parameters
    Flux.loadmodel!(net.best_model, net_state)
    Flux.testmode!(net.best_model)
    return net
end
    

# train_net(dgp)  # comment out when net is trained

net = load_trained()
data, θtrue = generate(dgp, net, 1)
θnn = net.best_model(data)[:]
@info "θtrue: ", θtrue
@info "θnn:", θnn
# set up proposal
covreps = 1000
_,Σₚ = simmomentscov(net, dgp, covreps, θnn)
δ = 1.0 # tuning

# do MCMC
S = 40 # simulations to compute moments
# initial short chain
chain = mcmc(θnn, θnn, δ, Σₚ, S, net, dgp; chainlength=500)
accept = mean(chain[:,end-1])
# loop to get good tuning
while accept < 0.2 || accept > 0.3
    accept < 0.2 ? δ *= 0.75 : nothing
    accept > 0.3 ? δ *= 1.5 : nothing
    chain = mcmc(θnn, θnn, δ, Σₚ, S, net, dgp; chainlength=500)
    accept = mean(chain[:,end-1])
end
# final long chain
chain = mcmc(θnn, θnn, δ, Σₚ, S, net, dgp; chainlength=5000)


