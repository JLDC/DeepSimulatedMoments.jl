export generate_files, load_from_file
"""
generate_files(dgp::AbstractDGP, dir::String, nfiles::Int; 
        batchsize::Int=1_024, verbosity::Int=(nfiles ÷ 10), 
        preprocessX::Function=identity, preprocessY::Function=identity
    )

Generate data files from a [`DGP`](@ref) and save it to a directory. Note that if 
the directory already contains files, the function will continue from the last 
file until the directory contains `nfiles`.

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
function generate_files(
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

"""
    load_from_file(file::String)

Load pre-generated data from a file.

# Arguments
- `file::String`: The file to load the data from.
"""
function load_from_file(file::String)
    file = jldopen(file, "r")
    @unpack X, Y = file
    X, Y
end

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
function RandomMixture(T::Type=Float32)
    RandomMixture{T}(1, 10, Uniform(-1, 1), Normal(0, 1), k -> rand(Dirichlet(k, 1)))
end

m


Random.rand(d::RandomMixture{T}) where {T<:AbstractFloat} = convert.(T, rand(make(d)))
Random.rand(d::RandomMixture{T}, n::Int) where {T<:AbstractFloat} = convert.(T, rand(make(d), n))




m = RandomMixture()



rand(m.μ)

m.μ

rand(m)


x = make(m)
rand(x, 10)

rand(m, 100)

rand(Normal(), 5)

rand(Normal{Float32}(0f0, 1f1))


x = Normal{Float32}(0f0, 1f0)
y = Normal()