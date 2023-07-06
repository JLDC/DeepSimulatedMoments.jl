export GARCH, nfeatures, nparams, θbounds, priordraw, generate, likelihood

"""
    GARCH{T} <: AbstractDGP{T}

A simple GARCH(1,1) process. (`T` defaults to `Float32`)

# Fields
- `N::Int`: Number of observations in each sample.
- `dist::ErrorDistribution`: Distribution of the error term.
"""
struct GARCH{T} <: AbstractDGP{T}
    N::Int
    dist::ErrorDistribution
end

"""
    GARCH(; N::Int, T::Type=Float32)

Create a `GARCH` DGP with `N` observations and `T` precision.

# Keyword Arguments
- `N::Int`: Number of observations in each sample.
- `T::Type`: Precision of the data (default: `Float32`, hint: `Float32` is
recommended for neural network compatibility).

# Returns
- `GARCH{T}`: GARCH DGP.
"""
GARCH(; N::Int, T::Type=Float32, dist=Normal(T(0), T(1))) = GARCH{T}(N, ErrorDistribution(dist))
GARCH(N::Int) = GARCH(N=N)

nfeatures(d::GARCH) = 1
nparams(d::GARCH) = 3

θbounds(::GARCH{T}) where T = (T[1f-4, 0, 0], T[1, 1 - 1f-1, 1])
function reparametrize(θ₁::T, θ₂::T, θ₃::T)::Tuple{T,T,T} where T 
    (1 - θ₂) * θ₁, (1 - θ₃) * θ₂, θ₂ * θ₃
end

# θ₁ is the long-run variance, θ₂ is the sum α + β, and θ₃ is the share of β
function simulate(d::GARCH{T}, θ₁::T, θ₂::T, θ₃::T) where T
    ω, α, β = reparametrize(θ₁, θ₂, θ₃)

    h, y = θ₁, zero(T)
    z = rand(d.dist, d.N)
    ys = zeros(T, d.N)

    @inbounds @simd for t ∈ eachindex(ys)
        h = ω + α * y ^ 2 + β * h
        y = √h * z[t]
        ys[t] = y
    end

    ys
end
simulate(d::GARCH{T}, θ::AbstractVector{T}) where T = simulate(d, θ...)

priordraw(d::GARCH{T}, S::Int) where T = uniformpriordraw(d, S)

@views function generate(d::GARCH{T}, S::Int) where T
    θ = priordraw(d, S)
    X = zeros(T, d.N, S)

    @inbounds Threads.@threads for s ∈ axes(X, 2)
        X[:, s] = simulate(d, θ[:, s]...)
    end
    
    permutedims(reshape(X, 1, d.N, S), (1, 3, 2)), θ
end

@views function generate(θ::AbstractVector{T}, d::GARCH{T}, S::Int) where T
    insupport(d, θ) || throw(ArgumentError("θ is not in the support of the prior"))
    X = zeros(T, d.N, S)

    @inbounds Threads.@threads for s ∈ axes(X, 2)
        X[:, s] = simulate(d, θ)
    end

    permutedims(reshape(X, 1, d.N, S), (1, 3, 2))
end

@views function likelihood(
    d::GARCH{T}, X::AbstractVector{T}, 
    θ₁::T, θ₂::T, θ₃::T
) where T
    ω, α, β = reparametrize(θ₁, θ₂, θ₃)
    h = zeros(T, d.N)
    X = X .^ 2
    h[1] = θ₁
    @inbounds for t ∈ 2:d.N
        h[t] = ω + α * X[t - 1] + β * h[t - 1]
    end

    # Return loglikelihood without constant part
    mean(-log.(h) .- X ./ h) / 2
end

likelihood(d::GARCH{T}, X::AbstractVector{T}, θ::AbstractVector{T}) where T = 
    likelihood(d, X, θ...)