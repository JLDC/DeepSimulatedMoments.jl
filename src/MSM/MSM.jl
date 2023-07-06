export run_msm

function msm_obj(
    θ̂ₓ::AbstractVector{T}, θ⁺::AbstractVector{T}; 
    mn::MomentNetwork, dgp::AbstractDGP{T}, S::Int,
    seed::Union{Nothing, Int}=nothing
) where {T<:AbstractFloat}
    # Ensure solution is in parameter space
    insupport(dgp, θ⁺) || return Inf
    isnothing(seed) || Random.seed!(seed)

    # Compute simulated moments and error between simulated and sample moments
    θ̂ₛ = simmoments(mn, dgp, S, θ⁺)
    sum(abs2, θ̂ₛ - θ̂ₓ)
end

function run_msm(
    X::AbstractArray{T,3}, dgp::AbstractDGP{T}, mn::MomentNetwork; S::Int
) where {T<:AbstractFloat}
    # Apply transforms to X
    Y = priordraw(dgp, 1) # Needed to apply transforms but can be ignored otherwise
    X, _ = apply_transforms(mn, X, Y)
    
    # Compute sample moments
    θ̂ₓ = make_moments(mn, X) |> vec
    seed = abs(rand(Int64)) # Set fixed seed for MSM objective

    # Compute MSM objective
    θ̂ₘₛₘ = optimize(θ⁺ -> msm_obj(
        θ̂ₓ, θ⁺, mn=mn, dgp=dgp, S=S, seed=seed), 
        θ̂ₓ, NelderMead()
    ).minimizer

    θ̂ₘₛₘ # MSM estimate
end