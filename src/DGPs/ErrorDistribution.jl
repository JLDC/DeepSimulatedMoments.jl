export RandomMixture, ErrorDistribution

"""
    RandomMixture{T<:AbstractFloat}

A Gaussian mixture model with random number of components.

# Fields
- `kmin::Int`: Minimum number of components.
- `kmax::Int`: Maximum number of components.
- `μ::Distribution`: Distribution of the mean of each component.
- `logσ::Distribution`: Distribution of the log standard deviation of each component.
- `fp::Function`: Function that draws the mixture weights from a Dirichlet distribution.
"""
struct RandomMixture{T<:AbstractFloat}
    kmin::Int
    kmax::Int
    μ::Distribution
    logσ::Distribution
    fp::Function
end

function make(d::RandomMixture)
    k = rand(d.kmin:d.kmax)
    MixtureModel([
        Normal(rand(d.μ), exp(rand(d.logσ))) for _ ∈ 1:k # TODO: This Float32/64 uniform thing is a bit annoying
    ], d.fp(k)) 
end
function RandomMixture(;
    kmin::Int=1, kmax::Int=20, 
    μ::Distribution=Uniform(-1, 1), logσ::Distribution=Normal(0, 0.5),
    fp::Function=k->rand(Dirichlet(k, 1)), T::Type=Float32
)
    RandomMixture{T}(kmin, kmax, μ, logσ, fp)
end

Random.rand(d::RandomMixture{T}) where {T<:AbstractFloat} = convert.(T, rand(make(d)))
Random.rand(d::RandomMixture{T}, n::Int) where {T<:AbstractFloat} = convert.(T, rand(make(d), n))


"""
    ErrorDistribution

A distribution for the error term in a [DGP](@ref AbstractDGP).
"""
struct ErrorDistribution
    dist::Union{Distribution, RandomMixture}
end


Random.rand(e::ErrorDistribution) = rand(e.dist)
Random.rand(e::ErrorDistribution, n::Int) = rand(e.dist, n)