module DeepSimulatedMoments

# using DifferentialEquations
using CUDA
using LinearAlgebra
using Flux
using StatsBase
using Distributions

include("DGPs/AbstractDGP.jl")
include("DGPs/MA2.jl")
include("DGPs/Logit.jl")
include("DGPs/GARCH.jl")
# include("DGPs/JumpDiffusion.jl")

include("NeuralNets/utils.jl")
include("NeuralNets/TCN.jl")
include("NeuralNets/tcn_utils.jl")

include("NeuralNets/HyperParameters.jl")
include("NeuralNets/MomentNetwork.jl")

include("MSM/MSMlib.jl")

export mcmc, getmoments, simmomentscov

end
