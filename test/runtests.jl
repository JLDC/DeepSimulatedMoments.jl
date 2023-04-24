using DeepSimulatedMoments
using Test

function test_dgp(d::AbstractDGP, S::Int)
    X, θ = generate(d, S)
    @test size(X) == (nfeatures(d), S, d.N)
    @test size(θ) == (nparams(d), S)
end

@testset "DeepSimulatedMoments.jl" begin
    N, S = [100 * 2^i for i ∈ 0:3], 25

    @testset "MA(2) checks" begin
        for n ∈ N
            test_dgp(MA2(n), S)
        end
    end

    @testset "Logit checks" begin
        for n ∈ N
            test_dgp(Logit(n), S)
        end
    end

    @testset "GARCH(1, 1) checks" begin
        for n ∈ N
            test_dgp(GARCH(n), S)
        end
    end
end
