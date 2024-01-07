using Distributions, Statistics

# computes neural moments from data
function getmoments(net, data)
    net.best_model(net.data_transform(data))
end    

# Compute simulated moments
function simmoments(
    mn::MomentNetwork, dgp::AbstractDGP{T}, S::Int, θ::AbstractVector{T}
) where {T<:AbstractFloat}
    data, _ = generate(θ, dgp, S)
    vec(mean(getmoments(mn, data), dims=2))
end

# Compute simulated moments and their covariance matrix
function simmomentscov(
    mn::MomentNetwork, dgp::AbstractDGP{T}, S::Int, θ::AbstractVector{T}
) where {T<:AbstractFloat}
    X = generate(θ, dgp, S)
    m = getmoments(mn, X)
    mean(m, dims=2), cov(permutedims(m))
end

# MVN random walk proposal
function proposal(
    x::AbstractVector{T}, δ::T, Σ::AbstractMatrix{T}
) where {T<:AbstractFloat}
rand(MvNormal(x, δ * Σ))
end

# CUE objective, in form of likelihood, to MAXIMIZE
function msm_obj(
    θ::AbstractVector{T}, # trial value
    θ̂ₓ::AbstractVector{T}, # real data net fit
    mn::MomentNetwork,
    dgp::AbstractDGP{T}, 
    S::Int 
) where {T<:AbstractFloat}
    !insupport(dgp,θ̂ₓ) ? error("data moment is not in prior suppor") : nothing  
    # Ensure solution is in parameter space
    insupport(dgp, θ) || return -Inf

    # Compute simulated moments and their covariance matrix
    θ̂ₛ, Σ̂ₛ = simmomentscov(mn, dgp, S, θ)
    Σ̂ₛ *= dgp.N*(1+1/S) # scale for accuracy
    isposdef(Σ̂ₛ) || return -Inf
    # Compute weighting matrix, return bmsm objective
    W = inv(Σ̂ₛ)
    err = √dgp.N * (θ̂ₛ - θ̂ₓ)
    -.5dot(err, W, err)
end



# TODO: prior? not needed at present, as priors are uniform
@views function mcmc(
    θ, # start value
    θ⁺, # the NN fitted value
    δ, # tuning
    Σₚ, # proposal covariance
    S,  # simulations per evaluation of likelihood
    mn, # MomentNetwork
    dgp;
    burnin::Int=100,
    chainlength::Int=1_000,
    verbosity::Int=10
)
    # set likelihood and proposal
    Lₙ = θ -> msm_obj(θ, θ⁺, mn, dgp, S) 
    proposal2 = θ -> proposal(θ, Float32(δ), Σₚ)

    Lₙθ = Lₙ(θ)
    naccept = 0 # Number of acceptance / rejections
    accept = false
    acceptance_rate = 1f0
    chain = zeros(chainlength, size(θ, 1) + 2)
    for i ∈ 1:burnin+chainlength
        θᵗ = proposal2(θ) # new trial value
        Lₙθᵗ = Lₙ(θᵗ) # Objective at trial value
        # Accept / reject trial value
        accept = rand() < exp(Lₙθᵗ - Lₙθ)
        if accept
            # Replace values
            θ = θᵗ
            Lₙθ = Lₙθᵗ
            # Increment number of accepted values
            naccept += 1
        end
        # Add to chain if burnin is passed
        # @info "current log-L" Lₙθ
        if i > burnin
            chain[i-burnin,:] = vcat(θ, accept, Lₙθ)
        end
        # Report
        if verbosity > 0 && mod(i, verbosity) == 0
            acceptance_rate = naccept / verbosity
            @info "Current parameters (iteration i=$i)" round.(θ, digits=3)' acceptance_rate
            naccept = 0
        end
    end
    return chain
end
