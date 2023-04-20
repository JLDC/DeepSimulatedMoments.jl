## Minimal interfaces to implement

To write a custom DGP, the following minimal interfaces must be implemented:
- [`nfeatures`](@ref), the number of features in the DGP
- [`nparams`](@ref), the number of parameters in the DGP
- [`priordraw`](@ref), the prior distribution of the parameters
- [`generate`](@ref), the function that generates a data set, with two methods:
    - [`generate(d::AbstractDGP, S::Int)`](@ref DeepSimulatedMoments.generate), the function that generates random parameters and a matching data set
    - [`generate(θ::AbstractVector{T}, d::AbstractDGP{T}, S::Int)`](@ref DeepSimulatedMoments.generate
)`](@ref), th), the function that generates a data set given a set of parameters `θ`

Once these are implemented, the user can train a neural network on the data set and proceed with statistical inference using the method of simulated moments. TODO: Provide links

## Example

The following example shows how to implement the minimal interfaces of a DGP for a simple linear regression model with `K` features.

#### 1. Define the DGP `struct`
```julia
struct LinearRegression{T} <: AbstractDGP{T}
    N::Int # Number of observations that our data set will have
    K::Int # Number of features in our data set
end
```

#### 2. Implement the `nfeatures` and `nparams` interface

```julia
nfeatures(d::LinearRegression) = d.K
nparams(d::LinearRegression) = d.K + 1 # We have K features and 1 intercept
```

#### 3. Implement the `priordraw` interface

```julia
function priordraw(d::LinearRegression{T}, S::Int) where T
    # Draw random parameters from a standard normal distribution 
    # (we could modify this to draw from a different distribution)
    θ = randn(T, S, nparams(d))
end
```

#### 4. Implement the `generate` interface

```julia
function generate(d::LinearRegression{T}, S::Int) where T
    # Draw random parameters from the prior
    θ = priordraw(d, S)
    # Generate a data set given the random parameters
    x = zeros(T, d.N, S)

end