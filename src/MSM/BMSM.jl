function bmsm_obj_cue(
    θ̂ₓ::AbstractVector{T}, θ⁺::AbstractVector{T}; 
    mn::MomentNetwork, dgp::AbstractDGP{T}, S::Int; 
) where {T<:AbstractFloat}
    # Ensure solution is in parameter space
    insupport(dgp, θ⁺) || return Inf

    # Compute simulated moments and their covariance matrix
    θ̂ₛ, Σ̂ₛ = simmomentscov(mn, dgp, S, θ⁺)
    
    # Compute weighting matrix, return bmsm objective
    W = inv(Σ̂ₛ)
    err = θ̂ₛ - θ̂ₓ
    dot(err, W, err)    
end

function bmsm_obj_2step(
    θ̂ₓ::AbstractVector{T}, θ⁺::AbstractVector{T}, Σ⁻¹::AbstractMatrix{T}; 
    mn::MomentNetwork, dgp::AbstractDGP{T}, S::Int; 
) where {T<:AbstractFloat}
    # Ensure solution is in parameter space
    insupport(dgp, θ⁺) || return Inf

    # Compute simulated moments only (no covariance matrix)
    θ̂ₛ = simmoments(mn, dgp, S, θ⁺)

    err = θ̂ₛ - θ̂ₓ
    dot(err, Σ⁻¹, err)
end