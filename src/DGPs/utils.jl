export generate_data
"""
    generate_data(dgp::AbstractDGP, dir::String, nfiles::Int; 
        batchsize::Int=1_024, verbosity::Int=(nfiles ÷ 10), 
        preprocessX::Function=identity, preprocessY::Function=identity
    )

Generate data from a [`DGP`](@ref) and save it to a directory. Note that if the 
directory already contains files, the function will continue from the last file
until the directory contains `nfiles`.

# Arguments
- `dgp::AbstractDGP`: The [`DGP`](@ref) to generate data from.
- `dir::String`: The directory to save the data to.
- `nfiles::Int`: The number of files to generate.

# Keyword Arguments
- `batchsize::Int`: The batch size (default: `1024`).
- `verbosity::Int`: The verbosity (default: `nfiles ÷ 10`).
- `preprocessX::Function`: The preprocessing function for the data (default: `identity`).
- `preprocessY::Function`: The preprocessing function for the parameters (default: `identity`).
"""
function generate_data(
    dgp::AbstractDGP, dir::String, nfiles::Int; 
    batchsize::Int=1_024, verbosity::Int=(nfiles ÷ 10),
    preprocessX::Function=identity, preprocessY::Function=identity
)
    # Make sure the directory exists, else create it
    isdir(dir) || mkdir(dir)

    # Check how many files the directory already contains
    files = readdir(dir)
    i = isempty(files) ? 0 : maximum(map(
        x -> parse(Int, split(x, ".")[1]), files
    ))

    newfiles = nfiles - i
    ndigits = length(string(nfiles))
    for f ∈ Base.OneTo(newfiles)
        i += 1 

        X, Y = generate(dgp, batchsize)
        X, Y = preprocessX(X), preprocessY(Y)

        jldsave(joinpath(dir, "$i.jld2"), X=X, Y=Y)

        if verbosity > 0 && f % verbosity == 0 
            @info Printf.format(Printf.Format("Generated %$(ndigits)d files."), i)
        end
    end
end