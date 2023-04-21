var documenterSearchIndex = {"docs":
[{"location":"api/#Data-Generating-Processes","page":"API","title":"Data Generating Processes","text":"","category":"section"},{"location":"api/#AbstractDGP","page":"API","title":"AbstractDGP","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"AbstractDGP\nnfeatures\nnparams\npriordraw\ngenerate","category":"page"},{"location":"api/#DeepSimulatedMoments.AbstractDGP","page":"API","title":"DeepSimulatedMoments.AbstractDGP","text":"AbstractDGP{T<:AbstractFloat}\n\nAbstract type for a DGP.\n\n\n\n\n\n","category":"type"},{"location":"api/#DeepSimulatedMoments.nfeatures","page":"API","title":"DeepSimulatedMoments.nfeatures","text":"nfeatures(d::AbstractDGP)\n\nNumber of features in the data.\n\nArguments\n\nd::AbstractDGP: DGP to get the number of features from.\n\nReturns\n\nInt: Number of features.\n\n\n\n\n\n","category":"function"},{"location":"api/#DeepSimulatedMoments.nparams","page":"API","title":"DeepSimulatedMoments.nparams","text":"nparams(d::AbstractDGP)\n\nNumber of parameters in the DGP.\n\nArguments\n\nd::AbstractDGP: DGP to get the number of parameters from.\n\nReturns\n\nInt: Number of parameters.\n\n\n\n\n\n","category":"function"},{"location":"api/#DeepSimulatedMoments.priordraw","page":"API","title":"DeepSimulatedMoments.priordraw","text":"priordraw(d::AbstractDGP, S::Int)\n\nDraw S parameter samples from the prior.\n\nArguments\n\nd::AbstractDGP{T}: DGP to draw the parameters from.\nS::Int: Number of samples to draw.\n\nReturns\n\nMatrix{T}: S samples of the parameters (dimension: S × nparams(d)).\n\n\n\n\n\n","category":"function"},{"location":"api/#DeepSimulatedMoments.generate","page":"API","title":"DeepSimulatedMoments.generate","text":"generate(d::AbstractDGP, S::Int)\n\nGenerate S data and parameter samples from the DGP.\n\nArguments\n\nd::AbstractDGP{T}: DGP to generate the data from.\nS::Int: Number of samples to generate.\n\nReturns\n\nTuple{Matrix{T}, Matrix{T}}: S samples of the data and parameters \n\n(dimension: nfeatures(d) × S × N and S × nparams(d)).\n\n\n\n\n\ngenerate(θ::AbstractVector{T}, d::AbstractDGP{T}, S::Int)\n\nGenerate S data samples from the DGP at parameters θ.\n\nArguments\n\nθ::AbstractVector{T}: Parameters of DGP used to generate the data.\nd::AbstractDGP{T}: DGP to generate the data from.\nS::Int: Number of samples to generate.\n\nReturns\n\nMatrix{T}: S samples of the data (dimension: nfeatures(d) × S × N).\n\n\n\n\n\n","category":"function"},{"location":"api/#Predefined-DGPs","page":"API","title":"Predefined DGPs","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"MA2\nLogit\nGARCH","category":"page"},{"location":"api/#DeepSimulatedMoments.MA2","page":"API","title":"DeepSimulatedMoments.MA2","text":"MA2{T} <: AbstractDGP{T}\n\nA simple MA(2) process. (T defaults to Float32)\n\nFields\n\nN::Int: Number of observations in each sample.\n\n\n\n\n\n","category":"type"},{"location":"api/#DeepSimulatedMoments.Logit","page":"API","title":"DeepSimulatedMoments.Logit","text":"Logit{T} <: AbstractDGP{T}\n\nA simple logistic regression model. (T defaults to Float32)\n\nFields\n\nN::Int: Number of observations in each sample.\nK::Int: Number of features in each sample.\n\n\n\n\n\n","category":"type"},{"location":"api/#DeepSimulatedMoments.GARCH","page":"API","title":"DeepSimulatedMoments.GARCH","text":"GARCH{T} <: AbstractDGP{T}\n\nA simple GARCH(1,1) process. (T defaults to Float32)\n\nFields\n\nN::Int: Number of observations in each sample.\n\n\n\n\n\n","category":"type"},{"location":"api/#Neural-Networks","page":"API","title":"Neural Networks","text":"","category":"section"},{"location":"api/#Temporal-Convolutional-Networks","page":"API","title":"Temporal Convolutional Networks","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"TemporalBlock\nTCN","category":"page"},{"location":"api/#DeepSimulatedMoments.TemporalBlock","page":"API","title":"DeepSimulatedMoments.TemporalBlock","text":"TemporalBlock(chan_in, chan_out; dilation, kernel_size, residual, pad, dropout_rate)\n\nTemporal block with chan_in input channels and chan_out output channels. Each block consists of two causal convolutional layers with kernel_size and dilation followed by batch normalization and dropout. If residual is true, a skip connection is added.\n\nArguments\n\nchan_in::Int: Number of input channels.\nchan_out::Int: Number of output channels.\n\nKeyword arguments\n\ndilation::Int: Kernel dilation.\nkernel_size::Int: Size of the convolutional kernel.\nresidual::Bool: Whether to use residual connections.\npad::Function: Padding to use for the convolutional layers.\ndropout_rate::AbstractFloat: Dropout rate to use for the convolutional layers.\n\nReturns\n\nChain: Temporal block\n\n\n\n\n\n","category":"function"},{"location":"api/#DeepSimulatedMoments.TCN","page":"API","title":"DeepSimulatedMoments.TCN","text":"TCN(channels; kernel_size, dilation_factor, residual, pad, dropout_rate)\n\nTemporal convolutional network (TCN) with length(channels) - 1 layers. Each layer is a TemporalBlock with channels[i] input channels and channels[i+1]\n\nArguments\n\nchannels::AbstractVector{Int}: Number of input and output channels for each layer.\nkernel_size::Int: Size of the convolutional kernel.\ndilation_factor::Int: Factor by which the dilation is increased for each layer. (default: 2)\nresidual::Bool: Whether to use residual connections. (default: true)\npad::Function: Padding to use for the convolutional layers. (default: SamePad())\ndropout_rate::AbstractFloat: Dropout rate to use for the convolutional layers. (default: 0.)\n\nReturns\n\nChain: TCN\n\n\n\n\n\n","category":"function"},{"location":"api/#Neural-Network-Utilities","page":"API","title":"Neural Network Utilities","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"tabular2conv\ntabular2rnn","category":"page"},{"location":"api/#DeepSimulatedMoments.tabular2conv","page":"API","title":"DeepSimulatedMoments.tabular2conv","text":"tabular2conv(X::AbstractArray{T, 3}) where T\n\nTransform a (K × S × T) array to a CNN format (1 × T × K × S array). K is the number of features, S is the number of samples, and T is the number of time steps / observations in each sample.\n\nArguments\n\nX::AbstractArray{T, 3}: Array to transform.\n\nReturns\n\nArray{T, 4}: CNN format of X.\n\n\n\n\n\n","category":"function"},{"location":"api/#DeepSimulatedMoments.tabular2rnn","page":"API","title":"DeepSimulatedMoments.tabular2rnn","text":"tabular2rnn(X::AbstractArray{T, 3}) where T\n\nTransform a (K × S × T) array to an RNN format (T-array of K × S matrices). K is the number of features, S is the number of samples, and T is the number  of time steps / observations in each sample.\n\nArguments\n\nX::AbstractArray{T, 3}: Array to transform.\n\nReturns\n\nVector{Matrix{T}}: RNN format of X.\n\n\n\n\n\n","category":"function"},{"location":"DGPs/#Data-Generating-Processes","page":"Data generating processes","title":"Data Generating Processes","text":"","category":"section"},{"location":"DGPs/","page":"Data generating processes","title":"Data generating processes","text":"Data Generating Processes (DGPs) are at the heart of our package. They are statistical models used to simulate data sets with specific characteristics, which makes them an essential component of the method of simulated moments.","category":"page"},{"location":"DGPs/","page":"Data generating processes","title":"Data generating processes","text":"To use DGPs in our package, you need to implement several minimal interfaces, which are described in the following section. Once you've implemented these interfaces, you can use the resulting DGP to generate data sets, train a neural network, and proceed with statistical inference using the method of simulated moments.","category":"page"},{"location":"DGPs/#Minimal-interfaces-to-implement","page":"Data generating processes","title":"Minimal interfaces to implement","text":"","category":"section"},{"location":"DGPs/","page":"Data generating processes","title":"Data generating processes","text":"To write a custom DGP, the following minimal interfaces must be implemented:","category":"page"},{"location":"DGPs/","page":"Data generating processes","title":"Data generating processes","text":"nfeatures(d::AbstractDGP), returns the number of features in the data set that the DGP generates, i.e., the dimension of one observation. Note that when a model has an outcome (e.g., a linear regression model), this outcome should count towards the number of features.\nnparams(d::AbstractDGP), returns the number of parameters in the DGP.\npriordraw(d::AbstractDGP, S::Int), returns a sample of the prior distribution of the parameters. The output corresponds to S parameter draws from the DGP d and is a matrix of size S × nparams(d).\ngenerate, the function that generates a data set, with two methods:\ngenerate(d::AbstractDGP, S::Int), generates random parameters and matching data sets. The first argument, d, should be an instance of the DGP. The second argument, S, should be an integer representing the number of data sets and parameter draws to be generated.\ngenerate(θ::AbstractVector{T}, d::AbstractDGP{T}, S::Int), generates S data sets given a set of parameters θ. The first argument, θ, should be a vector of parameter values in the format expected by the DGP. The second argument, d, should be an instance of the DGP. The third argument, S, should be an integer representing the number of data sets to be generated.","category":"page"},{"location":"DGPs/","page":"Data generating processes","title":"Data generating processes","text":"These interfaces provide the foundation for writing custom DGPs. With a well-defined DGP, you can train a neural network on the data set and proceed with statistical inference using the method of simulated moments.","category":"page"},{"location":"DGPs/#Example","page":"Data generating processes","title":"Example","text":"","category":"section"},{"location":"DGPs/","page":"Data generating processes","title":"Data generating processes","text":"This example shows how to implement a simple linear regression DGP with K features, which generates a data set with N observations.","category":"page"},{"location":"DGPs/#.-Define-the-DGP-struct","page":"Data generating processes","title":"1. Define the DGP struct","text":"","category":"section"},{"location":"DGPs/","page":"Data generating processes","title":"Data generating processes","text":"struct LinearRegression{T} <: AbstractDGP{T}\n    N::Int # Number of observations that our data set will have\n    K::Int # Number of features in our data set\nend","category":"page"},{"location":"DGPs/#.-Implement-the-nfeatures-and-nparams-interface","page":"Data generating processes","title":"2. Implement the nfeatures and nparams interface","text":"","category":"section"},{"location":"DGPs/","page":"Data generating processes","title":"Data generating processes","text":"nfeatures(d::LinearRegression) = d.K + 1 # We have K features and 1 outcome\nnparams(d::LinearRegression) = d.K + 1 # We have K features and 1 intercept","category":"page"},{"location":"DGPs/#.-Implement-the-priordraw-interface","page":"Data generating processes","title":"3. Implement the priordraw interface","text":"","category":"section"},{"location":"DGPs/","page":"Data generating processes","title":"Data generating processes","text":"function priordraw(d::LinearRegression{T}, S::Int) where T\n    # Draw random parameters from a standard normal distribution \n    # (we could modify this to draw from a different distribution)\n    θ = randn(T, S, nparams(d))\nend","category":"page"},{"location":"DGPs/#.-Implement-the-simulate-interface-(optional)","page":"Data generating processes","title":"3. Implement the simulate interface (optional)","text":"","category":"section"},{"location":"DGPs/","page":"Data generating processes","title":"Data generating processes","text":"While this interface is not necessary, it is a practical helper as we can use it in both implementations of the generate interface.","category":"page"},{"location":"DGPs/","page":"Data generating processes","title":"Data generating processes","text":"function simulate(d::LinearRegression{T}, θ::AbstractVector{T}) where T\n    ϵ = rand(T, d.N) # Error terms\n    # Features: one intercept and K random (standard normal) features\n    X = hcat(ones(T, d.N), randn(T, d.N, d.K))\n    # Outcomes\n    y = X * θ + ϵ\n    hcat(X, y)\nend","category":"page"},{"location":"DGPs/#.-Implement-the-generate-interfaces","page":"Data generating processes","title":"4. Implement the generate interfaces","text":"","category":"section"},{"location":"DGPs/","page":"Data generating processes","title":"Data generating processes","text":"# Generate S data sets and matching parameters\nfunction generate(d::LinearRegression{T}, S::Int) where T\n    # Draw random parameters from the prior\n    θ = priordraw(d, S)\n    X = zeros(T, d.N, S, nfeatures(d))\n\n    for s ∈ axes(X, 2)\n        X[:, s, :] = simulate(d, θ[s, :])\n    end\n\n    permutedims(X, (3, 2, 1)), θ\nend\n\n# Generate S data sets given a set of parameters θ\nfunction generate(θ::AbstractVector{T}, d::LinearRegression{T}, S::Int) where T\n    X = zeros(T, d.N, S, nfeatures(d))\n\n    for s ∈ axes(X, 2)\n        X[:, s, :] = simulate(d, θ)\n    end\n\n    permutedims(X, (3, 2, 1))\nend","category":"page"},{"location":"#DeepSimulatedMoments.jl","page":"Home","title":"DeepSimulatedMoments.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for DeepSimulatedMoments.jl","category":"page"}]
}
