struct Data{T}
    train_data::Vector{Tuple{Vector{T},Matrix{T}}}  # Each sample is a tuple of times and the corresponding observations
    val_data::Vector{Tuple{Vector{T},Matrix{T}}}
    test_data::Vector{Tuple{Vector{T},Matrix{T}}}
end

function Data{T}(train_data::SubArray, val_data::SubArray, test_data::SubArray) where {T}
    return Data{T}(copy(train_data), copy(val_data), copy(test_data))
end
