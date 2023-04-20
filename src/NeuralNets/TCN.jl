export TemporalBlock, TCN

"""
    TemporalBlock(chan_in, chan_out; dilation, kernel_size, residual, pad, dropout_rate)

Temporal block with `chan_in` input channels and `chan_out` output channels. Each
block consists of two causal convolutional layers with `kernel_size` and `dilation`
followed by batch normalization and dropout. If `residual` is `true`, a skip
connection is added.

# Arguments
- `chan_in::Int`: Number of input channels.
- `chan_out::Int`: Number of output channels.

# Keyword arguments
- `dilation::Int`: Kernel dilation.
- `kernel_size::Int`: Size of the convolutional kernel.
- `residual::Bool`: Whether to use residual connections.
- `pad::Function`: Padding to use for the convolutional layers.
- `dropout_rate::AbstractFloat`: Dropout rate to use for the convolutional layers.

# Returns
- `Chain`: Temporal block
"""
function TemporalBlock(
    chan_in::Int, chan_out::Int; 
    dilation::Int, kernel_size::Int,
    residual::Bool, pad::Function, dropout_rate::AbstractFloat
)
    # Causal convolutions
    causal_conv = Chain(
        Conv((1, kernel_size), chan_in => chan_out, dilation = dilation, 
            pad = pad),
        BatchNorm(chan_out, leakyrelu),
        Dropout(dropout_rate),
        Conv((1, kernel_size), chan_out => chan_out, dilation = dilation, 
            pad = pad),
        BatchNorm(chan_out, leakyrelu),
        Dropout(dropout_rate),
    )
    residual || return causal_conv
    # Skip connection (residual net)
    residual_conv = Conv((1, 1), chan_in => chan_out)
    Chain(
        Parallel(+, causal_conv, residual_conv),
        x -> leakyrelu.(x)
    )
end


"""
    TCN(channels; kernel_size, dilation_factor, residual, pad, dropout_rate)

Temporal convolutional network (TCN) with `length(channels) - 1` layers. Each layer
is a `TemporalBlock` with `channels[i]` input channels and `channels[i+1]`

# Arguments
- `channels::AbstractVector{Int}`: Number of input and output channels for each layer.
- `kernel_size::Int`: Size of the convolutional kernel.
- `dilation_factor::Int`: Factor by which the dilation is increased for each layer. (default: `2`)
- `residual::Bool`: Whether to use residual connections. (default: `true`)
- `pad::Function`: Padding to use for the convolutional layers. (default: `SamePad()`)
- `dropout_rate::AbstractFloat`: Dropout rate to use for the convolutional layers. (default: `0.`)

# Returns
- `Chain`: TCN
"""
function TCN(
    channels::AbstractVector{Int}; 
    kernel_size::Int, dilation_factor::Int=2, residual::Bool=true, 
    pad::Function=SamePad(), dropout_rate::AbstractFloat=0.,
)
    Chain([
        TemporalBlock(chan_in, chan_out, dilation=dilation_factor ^ (i - 1), 
            kernel_size=kernel_size, residual=residual, pad=pad,
            dropout_rate=dropout_rate
        ) 
        for (i, (chan_in, chan_out)) ∈ enumerate(zip(channels[1:end-1], channels[2:end]))]...)
end