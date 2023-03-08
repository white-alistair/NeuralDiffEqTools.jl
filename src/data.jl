abstract type AbstractData{T} end

struct TrainValidTestSplit{T} <: AbstractData{T}
    train_data::TimeSeries{T}
    valid_data::Array{TimeSeries{T}}
    test_data::Array{TimeSeries{T}}
end

function TrainValidTestSplit(
    time_series::TimeSeries{T},
    n_valid_sets::Int,
    valid_seconds::T;
    Δt::T,
) where {T}
    (; times, trajectory) = time_series

    total_valid_seconds = n_valid_sets * valid_seconds
    total_valid_obs = Int(total_valid_seconds / Δt) + 1

    train_data = TimeSeries{T}(
        times[1:end-total_valid_obs+1],
        trajectory[:, 1:end-total_valid_obs+1],
    )

    valid_data = TimeSeries{T}(
        times[end-total_valid_obs+1:end],
        trajectory[:, end-total_valid_obs+1:end],
    )

    chunked_valid_data = chunk(valid_data, Int(valid_seconds / Δt) + 1)

    return TrainValidTestSplit{T}(train_data, chunked_valid_data, [])
end

function TrainValidTestSplit(
    time_series::TimeSeries{T},
    n_valid_sets::Int,
    valid_seconds::T,
    n_test_sets::Int,
    test_seconds::T;
    Δt::T,
) where {T}
    (; times, trajectory) = time_series

    total_valid_seconds = n_valid_sets * valid_seconds
    valid_obs = Int(total_valid_seconds / Δt) + 1

    total_test_seconds = n_test_sets * test_seconds
    test_obs = Int(total_test_seconds / Δt) + 1

    test_index = size(times)[1] - test_obs + 1
    valid_index = test_index - valid_obs + 1

    train_data = TimeSeries{T}(times[1:valid_index], trajectory[:, 1:valid_index])

    valid_data =
        TimeSeries{T}(times[valid_index:test_index], trajectory[:, valid_index:test_index])

    test_data = TimeSeries{T}(times[test_index:end], trajectory[:, test_index:end])

    chunked_valid_data = chunk(valid_data, Int(valid_seconds / Δt) + 1)
    chunked_test_data = chunk(test_data, Int(test_seconds / Δt) + 1)

    return TrainValidTestSplit{T}(train_data, chunked_valid_data, chunked_test_data)
end

struct KLFolds{T} <: AbstractData{T}
    k::Int
    l::Int
    folds::Vector{TimeSeries{T}}
    test_folds::Vector{TimeSeries{T}}
end

function KLFolds(
    time_series::TimeSeries{T},
    k::Int,
    l::Int,
    n_test_folds::Int = 0;
    shuffle = true,
) where {T}
    total_folds = k + n_test_folds
    fold_obs = ceil(Int, length(time_series) / total_folds)
    folds = chunk(time_series, fold_obs)

    if shuffle
        shuffle!(folds)
    end

    if n_test_folds > 0
        folds, test_folds = folds[1:end-n_test_folds], folds[end-n_test_folds+1:end]
    else
        test_folds = []
    end

    return KLFolds{T}(k, l, folds, test_folds)
end

struct KLCycleIterator{T}
    kl_folds::KLFolds{T}
    epochs::Int
end

function kl_cycle(kl_folds::KLFolds, epochs::Int)
    return KLCycleIterator(kl_folds, epochs)
end

function Base.iterate(itr::KLCycleIterator)
    start_index = 1
    epoch = 1
    return iterate(itr, (start_index, epoch))
end

function Base.iterate(itr::KLCycleIterator, (start_index, epoch))
    if epoch > itr.epochs
        return nothing
    end

    (; k, l, folds) = itr.kl_folds

    end_index = start_index + l - 1
    if end_index > k
        start_index = 1
        end_index = l
    end

    validation_folds = @view folds[start_index:end_index]
    training_folds = @views [folds[1:start_index-1]; folds[end_index+1:end]]
    start_index = end_index + 1
    epoch += 1

    return (training_folds, validation_folds), (start_index, epoch)
end
