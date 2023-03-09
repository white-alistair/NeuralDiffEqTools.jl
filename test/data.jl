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

@testitem "KL Folds no test folds" begin
    Δt = 0.1
    times = collect(0.0:Δt:10.0)
    tr = rand(2, size(times)[1])
    time_series = NeuralDiffEqTools.TimeSeries{Float64}(times, tr)

    k = 10
    l = 2
    kl_folds = NeuralDiffEqTools.KLFold(time_series, k, l)

    @test size(kl_folds.folds)[1] == 10
end

@testitem "KL Folds with test folds" begin
    Δt = 0.1
    times = collect(0.0:Δt:12.0)
    tr = rand(2, size(times)[1])
    time_series = NeuralDiffEqTools.TimeSeries{Float64}(times, tr)

    n_test_folds = 2
    k = 10
    l = 2
    shuffle = true
    kl_folds = NeuralDiffEqTools.KLFold(time_series, k, l, n_test_folds; shuffle)

    @test size(kl_folds.test_folds) == (n_test_folds,)
    for fold in kl_folds.test_folds
        @test size(fold.times) == (11,)
        @test size(fold.trajectory) == (2, 11)        
    end
    @test size(kl_folds.folds) == (k,)
    for fold in kl_folds.folds
        @test size(fold.times) == (11,)
        @test size(fold.trajectory) == (2, 11)        
    end
end

@testitem "KL Folds cycle iterator" begin
    Δt = 0.1
    times = collect(0.0:Δt:0.8)
    tr = rand(2, size(times)[1])
    time_series = NeuralDiffEqTools.TimeSeries{Float64}(times, tr)

    k = 4
    l = 2
    kl_folds = NeuralDiffEqTools.KLFold(time_series, k, l; shuffle = false)

    epochs = 3
    itr = NeuralDiffEqTools.kl_cycle(kl_folds, epochs)

    (training_folds, validation_folds), state = iterate(itr)
    @test size(validation_folds) == (l,)
    @test size(training_folds) == (k - l,)
    @test validation_folds[1].times == times[1:3]
    @test validation_folds[1].trajectory == tr[:, 1:3]
    @test validation_folds[2].times == times[3:5]
    @test validation_folds[2].trajectory == tr[:, 3:5]
    (training_folds, validation_folds), state = iterate(itr, state)
    @test size(validation_folds) == (l,)
    @test size(training_folds) == (k - l,)
    @test validation_folds[1].times == times[5:7]
    @test validation_folds[1].trajectory == tr[:, 5:7]
    @test validation_folds[2].times == times[7:9]
    @test validation_folds[2].trajectory == tr[:, 7:9]
    (training_folds, validation_folds), state = iterate(itr, state)
    @test size(validation_folds) == (l,)
    @test size(training_folds) == (k - l,)
    @test validation_folds[1].times == times[1:3]
    @test validation_folds[1].trajectory == tr[:, 1:3]
    @test validation_folds[2].times == times[3:5]
    @test validation_folds[2].trajectory == tr[:, 3:5]
    @test iterate(itr, state) === nothing  # Finished
end
