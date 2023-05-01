using DeepSimulatedMoments
using Test

function test_dgp(d::AbstractDGP{T}, S::Int, test_likelihood::Bool=true) where T
    X, θ = generate(d, S)
    @test size(X) == (nfeatures(d), S, d.N)
    @test size(θ) == (nparams(d), S)
    if test_likelihood
        @test isa(likelihood(d, vec(X[:, 1, :]), θ[:, 1]), T)
    end
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
            test_dgp(Logit(n), S, false) # TODO
        end
    end

    @testset "GARCH(1, 1) checks" begin
        for n ∈ N
            test_dgp(GARCH(n), S)
        end
    end

    @testset "JumpDiffusion checks" begin
        for n ∈ N
            test_dgp(JumpDiffusion(n), S, false) # No lilkelihood
        end
    end
end
