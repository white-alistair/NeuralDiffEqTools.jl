function get_normalised_error(
    target::AbstractMatrix{T},
    pred::AbstractMatrix{T},
) where {T<:AbstractFloat}
    return vec(
        sqrt.(sum(abs2, target .- pred, dims = 1)) /
        sqrt(mean(sum(abs2, target, dims = 1))),
    )
end

function get_valid_time(
    target::AbstractMatrix{T},
    pred::AbstractMatrix{T},
    times::AbstractVector{T};
    error_threshold::T,
) where {T<:AbstractFloat}
    normalised_error = get_normalised_error(target, pred)
    return get_valid_time(normalised_error, times; error_threshold)
end

function get_valid_time(
    normalised_error::AbstractVector{T},
    times::AbstractVector{T};
    error_threshold::T,
) where {T<:AbstractFloat}
    valid_time_index = findfirst(error -> error > error_threshold, normalised_error)
    if isnothing(valid_time_index)
        valid_time_index = size(normalised_error)
    end
    return times[valid_time_index[1]]
end
