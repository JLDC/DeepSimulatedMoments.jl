export JumpDiffusion, nfeatures, nparams, θbounds, insupport, priordraw, generate
"""
    JumpDiffusion{T} <: AbstractDGP{T}

A jump diffusion stochastic volatility process. (`T` defaults to `Float32`).

# Fields
- `N::Int`: Number of observations in each sample.
"""
struct JumpDiffusion{T} <: AbstractDGP{T}
    N::Int
end

"""
    JumpDiffusion(; N::Int, T::Type=Float32)

Create a `JumpDiffusion` DGP with `N` observations and `T` precision.

# Keyword Arguments
- `N::Int`: Number of observations in each sample.
- `T::Type`: Precision of the data (default: `Float32`, hint: `Float32` is
recommended for neural network compatibility).

# Returns
- `JumpDiffusion{T}`: JumpDiffusion DGP.
"""
JumpDiffusion(; N::Int, T::Type=Float32) = JumpDiffusion{T}(N)
JumpDiffusion(N::Int) = JumpDiffusion(N=N)

nfeatures(d::JumpDiffusion) = 3
nparams(d::JumpDiffusion) = 8

θbounds(::JumpDiffusion{T}) where T = (
    #    μ,   κ,   α,   σ,    ρ,    λ₀, λ₁,    τ   
    T[-.05, .01,  -6, 0.1, -.99,  -.02,  3, -.02], 
    T[ .05, .30,   0, 4.0, -.50,   .05,  6,  .20]
)

function insupport(dgp::JumpDiffusion{T}, θ::AbstractVector{T}) where T
    lb, ub = θbounds(dgp)
    all(θ .>= lb) && all(θ .<= ub)
end

priordraw(d::JumpDiffusion, S::Int) = uniformpriordraw(d, S)

# For JumpDiffusion, due to the atom of probability at zero for λ₀ and τ, we
# need to use a different data transform than the default.
function datatransform(d::JumpDiffusion, S::Int; dev=cpu)
    pd = priordraw(d, S)
    pd[6, :] .= max.(pd[6, :], 0)
    pd[8, :] .= max.(pd[8, :], 0)
    fit(ZScoreTransform, dev(pd))
end


@views function simulate(
    d::JumpDiffusion{T}, θ::AbstractVector{T}; 
    burnin::Int=100, 
) where T

    trading_days = d.N # Number of trading days
    days = round(Int, 1.4 * (trading_days + burnin)) # Add weekends (x + x/5*2 = 1.4x)
    min_per_day = 1_440 # Minutes per day
    min_per_tic = 10 # Minutes between tics, lower for better accuracy
    tics = round(Int, min_per_day / min_per_tic) # Number of tics per day
    dt = 1/tics # Divisions per day
    closing = round(Int, 390 / min_per_tic) # Tic at closing (390 = 6.5 * 60)

    # Solve the diffusion
    μ, κ, α, σ, ρ, λ₀, λ₁, τ = θ
    # The prior allows for negative measurement error, to allow an accumulation at zero
    τ = max(0, τ) 
    u₀ = [μ; α]
    prob = diffusion(μ, κ, α, σ, ρ, u₀, (zero(T), days))
    λ₀⁺ = max(0, λ₀) # The prior allows for negative rate, to allow an accumulation at zero

    # # Jump in log price
    rate(u, p, t) = λ₀⁺

    # Jump is random sign time λ₁ times current std. dev.
    function affect!(integrator)
        integrator.u[1] = integrator.u[1] + rand([-1, 1]) * λ₁ * exp(integrator.u[2] / 2)
        nothing
    end

    jump = ConstantRateJump(rate, affect!)
    jump_prob = JumpProblem(prob, Direct(), jump)

    # Do the simulation
    sol = solve(jump_prob, SRIW1(), dt=dt, adaptive=false)

    # Get log price, with measurement error 
    # Trick: we only need very few log prices, 39 per trading day, use smart filtering
    lnPs = (
        T[sol(t)[1] + τ * randn() for t ∈ Iterators.take(p, closing)]
        for (_, p) ∈ Iterators.drop(
            Iterators.filter(
                x -> isweekday(x[1]), 
                enumerate(Iterators.partition(dt:dt:days, tics))), 
            burnin - 1)
    )

    # Get log price at end of trading days We will compute lag, so lose first
    lnP_trading = zeros(T, trading_days + 1)
    rv = zeros(T, trading_days + 1)
    bv = zeros(T, trading_days + 1) 

    p₋₁ = zero(T)
    @inbounds for (t, p) ∈ enumerate(lnPs)
        r = abs.(diff([p₋₁; p]))
        bv[t] = dot(r[2:end], r[1:end-1])
        rv[t] = dot(r[2:end], r[2:end])
        p₋₁ = p[end]
        lnP_trading[t] = p[end]
    end
    
    [diff(lnP_trading) rv[2:end] π .* bv[2:end] / 2]
end

@views function generate(d::JumpDiffusion{T}, S::Int) where T
    y = priordraw(d, S)
    x = zeros(T, d.N, 3, S)
    @inbounds Threads.@threads for s ∈ axes(x, 3)
        x[:, :, s] = simulate(d, y[:, s])
    end
    permutedims(x, (2, 3, 1)), y
end

@views function generate(θ::AbstractVector{T}, d::JumpDiffusion{T}, S::Int) where T
    x = zeros(T, d.N, 3, S)
    @inbounds Threads.@threads for s ∈ axes(x, 3)
        x[:, :, s] = simulate(d, θ)
    end
    permutedims(x, (2, 3, 1))
end

@views function generate(θ::AbstractMatrix{T}, d::JumpDiffusion{T}) where T
    x = zeros(T, d.N, 3, size(θ, 2))
    @inbounds Threads.@threads for s ∈ axes(x, 3)
        x[:, :, s] = simulate_jd(θ[:, s], d.N)
    end
    permutedims(x, (2, 3, 1))
end


# Utilities for simulation
isweekday(d::Int)::Bool = (d % 7) % 6 != 0

function diffusion(μ, κ, α, σ, ρ, u0, tspan)
    f = function(du,u,p,t)
        du[1] = μ # drift in log prices
        du[2] = κ .* (α .- u[2]) # mean reversion in shocks
    end
    g = function(du,u,p,t)
        du[1] = exp(u[2] / 2)
        du[2] = σ
    end
    noise = CorrelatedWienerProcess!([1 ρ; ρ 1], tspan[1], zeros(2), zeros(2))
    sde_f = SDEFunction{true}(f,g)
    SDEProblem(sde_f,g,u0,tspan,noise=noise)
end
