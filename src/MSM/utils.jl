# Compute simulated moments
function simmoments(
    mn::MomentNetwork, dgp::AbstractDGP{T}, S::Int, θ::AbstractVector{T}
) where {T<:AbstractFloat}
    X, _ = generate(θ, dgp, S, mn)
    vec(mean(make_moments(mn, X), dims=2))
end

# Compute simulated moments and their covariance matrix
function simmomentscov(
    mn::MomentNetwork, dgp::AbstractDGP{T}, S::Int, θ::AbstractVector{T}
) where {T<:AbstractFloat}
    X, _ = generate(θ, dgp, S, mn)
    m = make_moments(mn, X)
    mean(m, dims=2), cov(permutedims(m))
end

# MVN random walk proposal
function proposal(
    x::AbstractVector{T}, δ::T, Σ::AbstractMatrix{T}
) where {T<:AbstractFloat}
    x + δ * Σ * randn(T, size(x))
end