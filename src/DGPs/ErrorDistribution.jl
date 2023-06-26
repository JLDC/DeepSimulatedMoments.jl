export RandomMixture, ErrorDistribution

# Special random mixture for error terms
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
    μ::Distribution=Uniform(-1, 1), logσ::Distribution=Normal(0, 1),
    fp::Function=k->rand(Dirichlet(k, 1)), T::Type=Float32
)
    RandomMixture{T}(kmin, kmax, μ, logσ, fp)
end

Random.rand(d::RandomMixture{T}) where {T<:AbstractFloat} = convert.(T, rand(make(d)))
Random.rand(d::RandomMixture{T}, n::Int) where {T<:AbstractFloat} = convert.(T, rand(make(d), n))


struct ErrorDistribution
    dist::Union{Distribution, RandomMixture}
end


Random.rand(e::ErrorDistribution) = rand(e.dist)
Random.rand(e::ErrorDistribution, n::Int) = rand(e.dist, n)