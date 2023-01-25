@testset "valid_time.jl" begin
    @testset "Normalised error calc" begin
        ground_truth = [1.0 -1.0 1.0]
        predicted = [0.8 1.2 0.7]
        E = NeuralDiffEqTools.get_normalised_error(ground_truth, predicted)
        @test E ≈ [0.2, 2.2, 0.3]

        ground_truth = [1.0 1.0 1.0]
        predicted = [1.2 0.8 1.3]
        E = NeuralDiffEqTools.get_normalised_error(ground_truth, predicted)
        @test E ≈ [0.2, 0.2, 0.3]

        ground_truth = [
            3.0 4.0 3.0
            4.0 3.0 4.0
        ]
        predicted = [
            3.0 4.0 3.0
            4.5 4.0 4.0
        ]
        E = NeuralDiffEqTools.get_normalised_error(ground_truth, predicted)
        @test E ≈ [0.1, 0.2, 0.0]
    end

    @testset "Valid Time calc" begin
        normalised_error = [0.3, 0.3, 0.5, 0.6]
        times = [0.1, 0.2, 0.3, 0.4]
        valid_time =
            NeuralDiffEqTools.get_valid_time(normalised_error, times; valid_error_threshold = 0.4)
        @test valid_time ≈ 0.2

        normalised_error = [0.0, 0.3, 0.3]
        times = [0.0, 0.1, 0.2]
        valid_time =
            NeuralDiffEqTools.get_valid_time(normalised_error, times; valid_error_threshold = 0.4)
        @test valid_time ≈ 0.2

        normalised_error = [0.0, 0.4, 0.3]
        times = [0.0, 0.1, 0.2]
        valid_time =
            NeuralDiffEqTools.get_valid_time(normalised_error, times; valid_error_threshold = 0.4)
        @test valid_time ≈ 0.0
    end
end
