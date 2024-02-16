export ErrorDistribution
"""
    ErrorDistribution

A distribution for the error term in a [`AbstractDGP``](@ref).
"""
struct ErrorDistribution
    dist::Distribution
end


Random.rand(e::ErrorDistribution) = rand(e.dist)
Random.rand(e::ErrorDistribution, n::Int) = rand(e.dist, n)