export tabular2rnn, tabular2conv

"""
    tabular2rnn(X::AbstractArray{T, 3}) where T

Transform a (`K × S × T`) array to an RNN format (`T`-array of `K × S` matrices).
`K` is the number of features, `S` is the number of samples, and `T` is the number 
of time steps / observations in each sample.

# Arguments
- `X::AbstractArray{T, 3}`: Array to transform.

# Returns
- `Vector{Matrix{T}}`: RNN format of `X`.
"""
tabular2rnn(X::AbstractArray{T, 3}) where T = [view(X, :, :, i) for i ∈ axes(X, 3)]

"""
    tabular2conv(X::AbstractArray{T, 3}) where T

Transform a (`K × S × T`) array to a CNN format (`1 × T × K × S` array).
`K` is the number of features, `S` is the number of samples, and `T` is the number
of time steps / observations in each sample.

# Arguments
- `X::AbstractArray{T, 3}`: Array to transform.

# Returns
- `Array{T, 4}`: CNN format of `X`.
"""
@views tabular2conv(X) = permutedims(reshape(X, size(X)..., 1), (4, 3, 1, 2))


# In the following losses, Ŷ is always the sequence of predictions (RNN style prediction)
rmse_full(Ŷ, Y) = mean(sqrt, mean(mean(abs2.(ŷᵢ - Y) for ŷᵢ ∈ Ŷ), dims=2))
mse_full(Ŷ, Y) = mean(mean(abs2.(ŷᵢ - Y) for ŷᵢ ∈ Ŷ))
rmse_last(Ŷ, Y) = mean(sqrt, mean(abs2.(Ŷ[end] - Y), dims=2))
mse_last(Ŷ, Y) = mean(mean(abs2.(Ŷ[end] - Y)))

# In the following losses, Ŷ is the same dimension as Y (CNN style prediction)
rmse_conv(Ŷ, Y) = mean(sqrt.(mean(abs2.(Ŷ - Y), dims=2)))
mse_conv(Ŷ, Y) = mean(abs2, Ŷ - Y)