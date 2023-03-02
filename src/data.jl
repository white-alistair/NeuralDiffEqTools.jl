abstract type AbstractData{T} end

struct TrainValidTestSplit{T} <: AbstractData{T}
    train_data::TimeSeries{T}
    val_data::Array{TimeSeries{T}}
    test_data::Array{TimeSeries{T}}
end

# function chunk(array::Array, chunk_size::Int)
#     last_dim = ndims(array)
#     N = size(array)[end]
#     return [
#         selectdim(array, last_dim, i:min(i + chunk_size - 1, N)) for i = 1:chunk_size:N
#     ]
# end

function TrainValidTestSplit(
    time_series::TimeSeries{T},
    n_valid_sets::Int,
    valid_seconds::T;
    Δt::T,
) where {T}
    (; times, trajectory) = time_series

    total_valid_seconds = n_valid_sets * valid_seconds
    total_valid_obs = Int(total_valid_seconds / Δt) + 1

    train_data =
        TimeSeries{T}(times[1:end-total_valid_obs+1], trajectory[:, 1:end-total_valid_obs+1])

    valid_data =
        TimeSeries{T}(times[end-total_valid_obs+1:end], trajectory[:, end-total_valid_obs+1:end])
    
    chunked_valid_data = chunk(valid_data, Int(valid_seconds / Δt) + 1)

    return TrainValidTestSplit{T}(train_data, chunked_valid_data, [])
end
