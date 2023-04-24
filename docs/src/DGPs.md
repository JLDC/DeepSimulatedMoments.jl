# Data Generating Processes

Data Generating Processes (DGPs) are at the heart of our package. They are statistical models used to simulate data sets with specific characteristics, which makes them an essential component of the method of simulated moments.

To use DGPs in our package, you need to implement several minimal interfaces, which are described in the following section. Once you've implemented these interfaces, you can use the resulting DGP to generate data sets, train a neural network, and proceed with statistical inference using the method of simulated moments.

## Minimal interfaces to implement

To write a custom DGP, the following minimal interfaces must be implemented:
- [`nfeatures(d::AbstractDGP)`](@ref), returns the number of features in the data set that the DGP generates, i.e., the dimension of one observation. Note that when a model has an outcome (e.g., a linear regression model), this outcome should count towards the number of features.
- [`nparams(d::AbstractDGP)`](@ref), returns the number of parameters in the DGP.
- [`priordraw(d::AbstractDGP, S::Int)`](@ref), returns a sample of the prior distribution of the parameters. The output corresponds to `S` parameter draws from the DGP `d` and is a matrix of size `S × nparams(d)`.
- [`generate`](@ref), the function that generates a data set, with two methods:
    - [`generate(d::AbstractDGP, S::Int)`](@ref DeepSimulatedMoments.generate), generates random parameters and matching data sets. The first argument, `d`, should be an instance of the DGP. The second argument, `S`, should be an integer representing the number of data sets and parameter draws to be generated.
    - [`generate(θ::AbstractVector{T}, d::AbstractDGP{T}, S::Int)`](@ref DeepSimulatedMoments.generate), generates `S` data sets given a set of parameters `θ`. The first argument, `θ`, should be a vector of parameter values in the format expected by the DGP. The second argument, `d`, should be an instance of the DGP. The third argument, `S`, should be an integer representing the number of data sets to be generated.

These interfaces provide the foundation for writing custom DGPs. With a well-defined DGP, you can train a neural network on the data set and proceed with statistical inference using the method of simulated moments.

## Example

This example shows how to implement a simple linear regression DGP with `K` features, which generates a data set with `N` observations.

#### 1. Define the DGP `struct`
```julia
struct LinearRegression{T} <: AbstractDGP{T}
    N::Int # Number of observations that our data set will have
    K::Int # Number of features in our data set
end
```

#### 2. Implement the `nfeatures` and `nparams` interface

```julia
nfeatures(d::LinearRegression) = d.K + 1 # We have K features and 1 outcome
nparams(d::LinearRegression) = d.K + 1 # We have K features and 1 intercept
```

#### 3. Implement the `priordraw` interface

```julia
function priordraw(d::LinearRegression{T}, S::Int) where T
    # Draw random parameters from a standard normal distribution 
    # (we could modify this to draw from a different distribution)
    randn(T, S, nparams(d))
end
```

#### 4. Implement the `simulate` interface (optional)
While this interface is not necessary, it is a practical helper as we can use it in both implementations of the `generate` interface.

```julia
function simulate(d::LinearRegression{T}, θ::AbstractVector{T}) where T
    ϵ = rand(T, d.N) # Error terms
    # Features: one intercept and K random (standard normal) features
    X = hcat(ones(T, d.N), randn(T, d.N, d.K))
    # Outcomes
    y = X * θ + ϵ
    hcat(X, y)
end
```

#### 5. Implement the `generate` interfaces

```julia
# Generate S data sets and matching parameters
function generate(d::LinearRegression{T}, S::Int) where T
    # Draw random parameters from the prior
    θ = priordraw(d, S)
    X = zeros(T, d.N, S, nfeatures(d))

    for s ∈ axes(X, 2)
        X[:, s, :] = simulate(d, θ[s, :])
    end

    permutedims(X, (3, 2, 1)), θ
end

# Generate S data sets given a set of parameters θ
function generate(θ::AbstractVector{T}, d::LinearRegression{T}, S::Int) where T
    X = zeros(T, d.N, S, nfeatures(d))

    for s ∈ axes(X, 2)
        X[:, s, :] = simulate(d, θ)
    end

    permutedims(X, (3, 2, 1))
end
```