using DeepSimulatedMoments
using Test

@testset "DeepSimulatedMoments.jl" begin
    N, S = 100, 25
    dgp_testsets = Dict(
        "MA(2) checks" => MA2,
        "Logit checks" => Logit,
        "GARCH checks" => GARCH,
    )
    for (testset, DGP) in dgp_testsets
        @testset testset begin
            dgp = DGP(N)
            X, θ = generate(dgp, S)
            @test size(X) == (nfeatures(dgp), S, N)
            @test size(θ) == (S, nparams(dgp))
        end
    end
end
