module DeepSimulatedMoments

using CUDA
using Distributions
using Flux
using JLD2
using LinearAlgebra
using Optim
using Printf
using Random
using StatsBase


include("DGPs/AbstractDGP.jl")
include("DGPs/ErrorDistribution.jl")
include("DGPs/MA2.jl")
include("DGPs/Logit.jl")
include("DGPs/GARCH.jl")

include("NeuralNets/utils.jl")
include("NeuralNets/TCN.jl")
include("NeuralNets/tcn_utils.jl")

include("NeuralNets/HyperParameters.jl")
include("NeuralNets/MomentNetwork.jl")

include("MSM/MSMlib.jl")

export mcmc, getmoments, simmomentscov

end
