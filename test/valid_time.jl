@testset "valid_time.jl" begin
    normalised_error = [0.3, 0.3, 0.5, 0.6]
    times = [0.1, 0.2, 0.3, 0.4]
    valid_time =
        NeuralDiffEqTools.get_valid_time(normalised_error, times; error_threshold = 0.4)
    @test valid_time ≈ 0.2

    normalised_error = [0.0, 0.3, 0.3]
    times = [0.0, 0.1, 0.2]
    valid_time =
        NeuralDiffEqTools.get_valid_time(normalised_error, times; error_threshold = 0.4)
    @test valid_time ≈ 0.2

    normalised_error = [0.0, 0.4, 0.3]
    times = [0.0, 0.1, 0.2]
    valid_time =
        NeuralDiffEqTools.get_valid_time(normalised_error, times; error_threshold = 0.4)
    @test valid_time ≈ 0.0
end
