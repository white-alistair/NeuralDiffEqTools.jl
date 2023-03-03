@testitem "Train-validation split" begin
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

@testitem "Train-validation-test Split" begin
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

@testitem "KL Folds" begin
    Δt = 0.01
    times = collect(0.0:Δt:100.0)
    tr = rand(2, size(times)[1])
    time_series = NeuralDiffEqTools.TimeSeries{Float64}(times, tr)

    kl_folds = NeuralDiffEqTools.KLFolds(time_series, 10, 2)

    @test size(kl_folds.folds)[1] == 10
end

@testitem "KL Folds iteration interface" begin
    Δt = 0.1
    times = collect(0.0:Δt:1.2)
    tr = rand(2, size(times)[1])
    time_series = NeuralDiffEqTools.TimeSeries{Float64}(times, tr)

    k = 6
    l = 2
    kl_folds = NeuralDiffEqTools.KLFolds(time_series, k, l; shuffle = false)

    (training_folds, validation_folds), start_index = iterate(kl_folds)
    @test size(validation_folds) == (l,)
    @test size(training_folds) == (k - l,)
    @test validation_folds[1].times ≈ times[1:3]
    @test validation_folds[2].times ≈ times[3:5]
    (training_folds, validation_folds), start_index = iterate(kl_folds, start_index)
    @test size(validation_folds) == (l,)
    @test size(training_folds) == (k - l,)
    @test validation_folds[1].times == times[5:7]
    @test validation_folds[2].times == times[7:9]
    (training_folds, validation_folds), start_index = iterate(kl_folds, start_index)
    @test validation_folds[1].times == times[9:11]
    @test validation_folds[2].times == times[11:13]
    (training_folds, validation_folds), start_index = iterate(kl_folds, start_index)
    @test validation_folds[1].times == times[1:3]
    @test validation_folds[2].times == times[3:5]
end
