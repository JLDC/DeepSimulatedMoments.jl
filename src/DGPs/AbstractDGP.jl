export AbstractDGP, priordraw, generate, nfeatures, nparams, θbounds, 
    datatransform, likelihood

"""
    AbstractDGP{T<:AbstractFloat}

Abstract type for a DGP.
"""
abstract type AbstractDGP{T<:AbstractFloat} end

error_msg(t, f) = error("$f not implemented for ::$t")

# ----- Generic functions for DGPs -----

"""
    nfeatures(d::AbstractDGP)

Number of features in the data.

# Arguments
- `d::AbstractDGP`: DGP to get the number of features from.

# Returns
- `Int`: Number of features.
"""
nfeatures(d::AbstractDGP) = error_msg(typeof(d), "nfeatures")

"""
    nparams(d::AbstractDGP)

Number of parameters in the DGP.

# Arguments
- `d::AbstractDGP`: DGP to get the number of parameters from.

# Returns
- `Int`: Number of parameters.
"""
nparams(d::AbstractDGP) = error_msg(typeof(d), "nparams")
θbounds(d::AbstractDGP, args...) = error_msg(typeof(d), "θbounds")
function insupport(d::AbstractDGP{T}, θ::AbstractVector{T}) where T
    lb, ub = θbounds(d)
    all(θ .≥ lb) && all(θ .≤ ub) 
end

"""
    priordraw(d::AbstractDGP, S::Int)

Draw `S` parameter samples from the prior.

# Arguments
- `d::AbstractDGP{T}`: DGP to draw the parameters from.
- `S::Int`: Number of samples to draw.

# Returns
- `Matrix{T}`: `S` samples of the parameters (dimension: `S × nparams(d)`).
"""
priordraw(d::AbstractDGP, S::Int) = error_msg(typeof(d), "priordraw")
# Generate prior parameters according to uniform with lower and upper bounds
function uniformpriordraw(d::AbstractDGP{T}, S::Int) where T
    lb, ub = θbounds(d)
    (ub .- lb) .* rand(T, size(lb, 1), S) .+ lb 
end
# Data transform for a particular DGP
datatransform(d::AbstractDGP, S::Int; dev=cpu) = fit(ZScoreTransform, dev(priordraw(d, S)))

"""
    generate(d::AbstractDGP, S::Int)

Generate `S` data and parameter samples from the DGP.

# Arguments
- `d::AbstractDGP{T}`: DGP to generate the data from.
- `S::Int`: Number of samples to generate.

# Returns
- Tuple{`Matrix{T}`, `Matrix{T}`}: `S` samples of the data and parameters 
(dimension: `nfeatures(d) × S × N` and `S × nparams(d)`).
"""
generate(d::AbstractDGP, S::Int) = error_msg(typeof(d), "generate")
# Generate to specific device directly
generate(d::AbstractDGP, S::Int; dev=cpu) = map(dev, generate(d, S))

"""
    generate(θ::AbstractVector{T}, d::AbstractDGP{T}, S::Int)

Generate `S` data samples from the DGP at parameters `θ`.

# Arguments
- `θ::AbstractVector{T}`: Parameters of DGP used to generate the data.
- `d::AbstractDGP{T}`: DGP to generate the data from.
- `S::Int`: Number of samples to generate.

# Returns
- `Matrix{T}`: `S` samples of the data (dimension: `nfeatures(d) × S × N`).
"""
generate(θ::AbstractVector{T}, d::AbstractDGP{T}, S::Int) where T = 
    error_msg(typeof(d), "generate")

"""
    likelihood(d::AbstractDGP, X::AbstractArray, θ::AbstractVector)

Compute the log-likelihood of the data `X` given the parameters `θ`. This 
function is irrelevant for DGPs with an intractable likelihood.

# Arguments
- `d::AbstractDGP{T}`: DGP to compute the log-likelihood for.
- `X::AbstractArray{T}`: Data to compute the log-likelihood for.
- `θ::AbstractVector{T}`: Parameters to compute the log-likelihood for.

# Returns
- `T`: Log-likelihood of the data given the parameters.
"""
likelihood(d::AbstractDGP{T}, X::AbstractArray{T}, θ::AbstractVector{T}) where T = 
    error_msg(typeof(d), "likelihood")


# Expected absolute error when using the prior mean as prediction 
# θbounds has to be defined => uniform priors only
priorerror(d::AbstractDGP) = .25abs.(reduce(-, θbounds(d)))
priorpred(d::AbstractDGP) = .5reduce(+, θbounds(d))