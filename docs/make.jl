using Documenter
using DeepSimulatedMoments

makedocs(
    sitename = "DeepSimulatedMoments",
    format = Documenter.HTML(),
    modules = [DeepSimulatedMoments],
    pages = [
        "Home" => "index.md",
        "Data generating processes" => "DGPs.md",
        "API" => "api.md",
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/JLDC/DeepSimulatedMoments.jl"
)
