module DeepSimulatedMoments

using DifferentialEquations
using Distributions
using LinearAlgebra
using Flux
using JLD2
using Optim
using Printf
using Random
using StatsBase
using UnPack

include("DGPs/AbstractDGP.jl")
include("DGPs/ErrorDistribution.jl")
include("DGPs/MA2.jl")
include("DGPs/Logit.jl")
include("DGPs/GARCH.jl")
include("DGPs/JumpDiffusion.jl")
include("DGPs/utils.jl")

include("NeuralNets/utils.jl")
include("NeuralNets/TCN.jl")
include("NeuralNets/tcn_utils.jl")

include("NeuralNets/HyperParameters.jl")
include("NeuralNets/MomentNetwork.jl")

# include("MSM/utils.jl")
# include("MSM/BMSM.jl")
# include("MSM/MSM.jl")

end