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
necessary_layers(dilation::Int, kernel_size::Int, receptive_field::Int) =
    log(dilation, (receptive_field - 1) * (dilation - 1) / (kernel_size - 1)) + 1

function necessary_layers(
    dilation::Int, kernel_size::Int, receptive_field_size::Int; use_ceil::Bool
) 
    necessary_layers(dilation, kernel_size, receptive_field_size) |> 
    x -> use_ceil ? ceil(Int, x) : x
end

function necessary_layers(
    dilation::Int, kernel_size::Int, receptive_field_size::Int; use_floor::Bool, 
)
    necessary_layers(dilation, kernel_size, receptive_field_size) |> 
    x -> use_floor ? floor(Int, x) : x
end

