export build_tcn, receptive_field_size, necessary_layers

"""
    receptive_field_size(dilation, kernel_size, layers)

Computes the receptive field size for a specified dilation, kernel size, and 
number of layers.

# Arguments
- `dilation::Int`: Dilation factor.
- `kernel_size::Int`: Size of the convolutional kernel.
- `layers::Int`: Number of layers.

# Returns
- `Int`: Receptive field size.
"""
receptive_field_size(dilation::Int, kernel_size::Int, layers::Int) = 
    1 + (kernel_size - 1) * (dilation ^ layers - 1) / (dilation - 1)

"""
    necessary_layers(dilation, kernel_size, receptive_field)

Computes the minimum number of layers necessary to achieve a specified receptive
field size.

# Arguments
- `dilation::Int`: Dilation factor.
- `kernel_size::Int`: Size of the convolutional kernel.
- `receptive_field::Int`: Desired receptive field size.

# Keyword Arguments
- `use_ceil::Bool`: Whether to use `ceil(Int, x)` for the output. (default: `false`)
- `use_floor::Bool`: Whether to use `floor(Int, x)` for the output. (default: `false`)

# Returns
- Minimum number of layers (without `use_ceil` or `use_floor`, this is a `Float64`).
"""
function necessary_layers(
    dilation::Int, kernel_size::Int, receptive_field::Int;
    use_ceil::Bool=false, use_floor::Bool=false
)
    use_ceil && use_floor && throw(
        ArgumentError("Cannot use both `use_ceil` and `use_floor`"))
    
    n = log(dilation, (receptive_field - 1) * (dilation - 1) / (kernel_size - 1)) + 1
    use_ceil ? ceil(Int, n) : use_floor ? floor(Int, n) : n
end

"""
    build_tcn(d::AbstractDGP; dilation_factor, kernel_size, channels, 
        summary_size, residual, pad)
    
Builds a TCN for a specified DGP.

# Arguments
- `d::AbstractDGP`: DGP to build the TCN for.

# Keyword Arguments
- `nlayers::Int`: Number of layers. (default: `0`, computes the necessary 
    layers using [`receptive_field_size`](@ref))
- `dilation_factor::Int`: Dilation factor. (default: `2`)
- `kernel_size::Int`: Size of the convolutional kernel. (default: `8`)
- `channels::Int`: Number of channels. (default: `32`)
- `receptive_field_size::Int`: Receptive field size. (default: `0`, uses the 
    length of the [`AbstractDGP`](@ref) instead)
- `summary_size::Int`: Size of the summary convolution. (default: `10`)
- `residual::Bool`: Whether to use residual connections. (default: `true`)
- `pad`: Padding function. (default: `SamePad()`)
- `dev::Function`: Device to use. (default: `cpu`)

# Returns
- `Chain`: TCN for the specified DGP.
"""
function build_tcn(
    d::AbstractDGP; 
    nlayers::Int=0, dilation_factor::Int=2, kernel_size::Int=8, channels::Int=32, 
    receptive_field_size::Int=0, summary_size::Int=10, residual::Bool=true,  
    pad=SamePad(), dev::Function=cpu
)

    if nlayers == 0
        receptive_field_size = receptive_field_size == 0 ? d.N : receptive_field_size
        nlayers = necessary_layers(dilation_factor, kernel_size, receptive_field_size,
            use_ceil=true)
    end

    dim_in, dim_out = nfeatures(d), nparams(d)
    dev(
        Chain(
            TCN(
                vcat(dim_in, [channels for _ in 1:nlayers], 1),
                kernel_size=kernel_size, pad=pad, residual=residual, 
                dilation_factor=dilation_factor
            ),
            Conv((1, summary_size), 1 => 1, stride=summary_size),
            Flux.flatten,
            Dense(d.N ÷ summary_size => d.N ÷ summary_size, hardtanh), # this is a new layer
            Dense(d.N ÷ summary_size => dim_out)
        )
    )
end