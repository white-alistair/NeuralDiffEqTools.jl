@testset "data.jl" begin
    @testset "Train Validation Split" begin
        Δt = 0.01
        times = collect(0.0:Δt:100.0)
        tr = rand(2, size(times)[1])
        time_series = NeuralDiffEqTools.TimeSeries{Float64}(times, tr)

        (; train_data, valid_data) =
            NeuralDiffEqTools.TrainValidTestSplit(time_series, 2, 4.0; Δt)

        @test train_data.times[end] == 92.0
        @test valid_data[1].times[1] == 92.0
        @test valid_data[1].times[end] == 96.0
        @test valid_data[2].times[1] == 96.0
        @test valid_data[2].times[end] == 100.0
    end

    @testset "Train Validation Test Split" begin
        Δt = 0.01
        times = collect(0.0:Δt:100.0)
        tr = rand(2, size(times)[1])
        time_series = NeuralDiffEqTools.TimeSeries{Float64}(times, tr)

        (; train_data, valid_data, test_data) =
            NeuralDiffEqTools.TrainValidTestSplit(time_series, 2, 4.0, 3, 5.0; Δt)

        @test train_data.times[end] == 77.0
        @test valid_data[1].times[1] == 77.0
        @test valid_data[1].times[end] == 81.0
        @test valid_data[2].times[1] == 81.0
        @test valid_data[2].times[end] == 85.0
        @test test_data[1].times[1] == 85.0
        @test test_data[1].times[end] == 90.0
        @test test_data[2].times[1] == 90.0
        @test test_data[2].times[end] == 95.0
        @test test_data[3].times[1] == 95.0
        @test test_data[3].times[end] == 100.0
    end

    @testset "KL Folds" begin
        Δt = 0.01
        times = collect(0.0:Δt:100.0)
        tr = rand(2, size(times)[1])
        time_series = NeuralDiffEqTools.TimeSeries{Float64}(times, tr)

        kl_folds =
            NeuralDiffEqTools.KLFolds(time_series, 10, 2)

        @test size(kl_folds.folds)[1] == 10
    end
end
