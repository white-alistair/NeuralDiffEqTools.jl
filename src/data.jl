struct Data{T}
    train_data::Vector{Tuple{Vector{T},Matrix{T}}}  # Each sample is a tuple of times and the corresponding observations
    val_data::Vector{Tuple{Vector{T},Matrix{T}}}
    test_data::Vector{Tuple{Vector{T},Matrix{T}}}
end

function Data{T}(train_data::SubArray, val_data::SubArray, test_data::SubArray) where {T}
    return Data{T}(copy(train_data), copy(val_data), copy(test_data))
end

function Data{T}(train_data, val_data) where {T}
    return Data{T}(train_data, val_data, [])
end

function Data{T}(
    time_series::TimeSeries{T};
    steps::Int,
    split_at,
    shuffle = false,
) where {T}
    data_ms = multiple_shooting(time_series, steps)
    data_split = MLUtils.splitobs(data_ms; at = split_at, shuffle)
    return Data{T}(data_split...)
end
