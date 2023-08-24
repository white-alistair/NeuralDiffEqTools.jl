@testitem "TimeSeries" begin
    # just test that the constructors work correctly
    using OrdinaryDiffEq 

    f(u,p,t) = -0.9 .* u
    prob = ODEProblem(f, rand(5), (0.,1.))
    sol = solve(prob, Tsit5())

    time_series = NeuralDiffEqTools.TimeSeries{Float64}(sol)
    @test typeof(time_series) <: NeuralDiffEqTools.TimeSeries 

    time_series = NeuralDiffEqTools.TimeSeries(sol.t, Array(sol))
    @test typeof(time_series) <: NeuralDiffEqTools.TimeSeries 

end