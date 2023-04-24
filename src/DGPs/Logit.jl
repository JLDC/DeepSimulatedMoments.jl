export Logit, nfeatures, nparams, priordraw, generate, likelihood

"""
    Logit{T} <: AbstractDGP{T}

A simple logistic regression model. (`T` defaults to `Float32`)

# Fields
- `N::Int`: Number of observations in each sample.
- `K::Int`: Number of features in each sample.
"""
struct Logit{T} <: AbstractDGP{T}
    N::Int
    K::Int
end

"""
    Logit(; N::Int=100, K::Int=3, T::Type=Float32)

Create a `Logit` DGP with `N` observations, `K` features, and `T` precision.

# Arguments
- `N::Int`: Number of observations in each sample.
- `K::Int`: Number of features in each sample.
- `T::Type`: Precision of the data (default: `Float32`, hint: `Float32` is
recommended for neural network compatibility).

# Returns
- `Logit{T}`: Logit DGP.
"""
Logit(; N::Int=100, K::Int=3, T::Type=Float32) = Logit{T}(N, K)
Logit(N::Int, K::Int=3) = Logit(N=N, K=K)

nfeatures(d::Logit) = d.K + 1 # +1 for the response
nparams(d::Logit) = d.K # No intercept

priordraw(d::Logit{T}, S::Int) where T = rand(T, d.K, S)

function simulate(d::Logit{T}, θ::AbstractVector{T}) where T
    ϵ = rand(T, d.N)
    X = randn(T, d.N, d.K)
    y = ϵ .< 1 ./ (1 .+ exp.(-X * θ))
    hcat(X, y)
end

priordraw(d::Logit{T}, S::Int) where T = rand(T, d.K, S)

@views function generate(d::Logit{T}, S::Int) where T
    θ = priordraw(d, S)
    X = zeros(T, d.N, S, nfeatures(d))

    @inbounds Threads.@threads for s ∈ axes(X, 2)
        X[:, s, :] = simulate(d, θ[:, s])
    end
    
    permutedims(X, (3, 2, 1)), θ
end

@views function generate(θ::AbstractVector{T}, d::Logit{T}, S::Int) where T
    X = zeros(T, d.N, S, nfeatures(d))

    @inbounds Threads.@threads for s ∈ axes(X, 2)
        X[:, s, :] = simulate(d, θ)
    end
    
    permutedims(X, (3, 2, 1))
end