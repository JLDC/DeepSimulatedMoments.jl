export MA2, nfeatures, nparams, θbounds, insupport, priordraw, generate, likelihood

"""
    MA2{T} <: AbstractDGP{T}

A simple MA(2) process. (`T` defaults to `Float32`)

# Fields
- `N::Int`: Number of observations in each sample.
"""
struct MA2{T} <: AbstractDGP{T}
    N::Int
    dist::ErrorDistribution
end

"""
    MA2(; N::Int, T::Type=Float32)

Create a `MA2` DGP with `N` observations and `T` precision.

# Keyword Arguments
- `N::Int`: Number of observations in each sample.
- `T::Type`: Precision of the data (default: `Float32`, hint: `Float32` is 
recommended for neural network compatibility).

# Returns
- `MA2{T}`: MA(2) DGP.
"""
MA2(; N::Int, T::Type=Float32, dist=Normal(T(0), T(1))) = MA2{T}(N, ErrorDistribution(dist))
MA2(N::Int) = MA2(N=N)

nfeatures(d::MA2) = 1
nparams(d::MA2) = 2

θbounds(::MA2{T}) where T = (T[-2, -1], T[2, 1])
# Overload insupport function for triangular support
insupport(::MA2{T}, θ₁::T, θ₂::T) where T = 
    (-2 < θ₁ < 2) && (-1 < θ₂ < 1) && (θ₂ - abs(θ₁) > -1)
insupport(d::MA2{T}, θ::AbstractVector{T}) where T = insupport(d, θ...)

function simulate(d::MA2{T}, θ₁::T, θ₂::T) where T
    ϵ = rand(d.dist, d.N + 2)
    @. ϵ[3:end] + θ₁ * ϵ[2:end-1] + θ₂ * ϵ[1:end-2]
end
simulate(d::MA2{T}, θ::AbstractVector{T}) where T = simulate(d, θ...)

# Use rejection sampling to stay inside identified region
@views function priordraw(d::MA2{T}, S::Int) where T
    θ = zeros(T, 2, S)
    Threads.@threads for s ∈ axes(θ, 2)
        ok = false
        θ₁, θ₂ = zero(T), zero(T)
        while !ok
            θ₁ = 4rand(T) - 2
            θ₂ = 2rand(T) - 1
            ok = insupport(d, θ₁, θ₂)
        end
        θ[:, s] = [θ₁, θ₂]
    end
    θ
end

@views function generate(d::MA2{T}, S::Int) where T
    θ = priordraw(d, S)
    X = zeros(T, d.N, S)
    
    @inbounds Threads.@threads for s ∈ axes(X, 2)
        X[:, s] = simulate(d, θ[:, s])
    end

    permutedims(reshape(X, 1, d.N, S), (1, 3, 2)), θ
end

@views function generate(θ::AbstractVector{T}, d::MA2{T}, S::Int) where T
    insupport(d, θ) || throw(ArgumentError("θ is not in support"))
    X = zeros(T, d.N, S)
    
    @inbounds Threads.@threads for s ∈ axes(X, 2)
        X[:, s] = simulate(d, θ)
    end

    permutedims(reshape(X, 1, d.N, S), (1, 3, 2))
end

@views function likelihood(
    d::MA2{T}, X::AbstractVector{T}, θ::AbstractVector{T}, 
    Σ::AbstractMatrix{T}, Σ⁻¹::AbstractMatrix{T}
) where T
    insupport(d, θ) || return -Inf
    n = size(X, 1)
    Σ!(θ, Σ)
    Σ⁻¹ .= inv(Σ)

    # Return loglikelihood without constant part
    -log(det(Σ)) / 2n - dot(X, Σ⁻¹, X)
end

function likelihood(d::MA2{T}, X::AbstractVector{T}, θ::AbstractVector{T}) where T
    n = size(X, 1)
    Σ = zeros(T, n, n)
    Σ⁻¹ = zeros(T, n, n)
    likelihood(d, X, θ, Σ, Σ⁻¹)
end

# Fill in the covariance matrix for MA2
@inbounds @views function Σ!(θ::AbstractVector{T}, Σ::AbstractMatrix{T}) where T
    n = size(Σ,1)
    ϕ₁, ϕ₂ = θ
    for t ∈ 1:n
        Σ[t, t] = 1 + ϕ₁ ^ 2 + ϕ₂ ^ 2
    end
    for t ∈ 1:n-1
        Σ[t, t+1] = ϕ₁ * (1 + ϕ₂)
        Σ[t+1, t] = ϕ₁ * (1 + ϕ₂)
    end
    for t ∈ 1:n-2
        Σ[t, t+2] = ϕ₂ 
        Σ[t+2, t] = ϕ₂
    end
end
