function get_normalised_error(
    ground_truth::AbstractMatrix{T},
    predicted::AbstractMatrix{T},
) where {T<:AbstractFloat}
    return vec(
        sqrt.(sum(abs2, ground_truth .- predicted; dims = 1)) /
        sqrt(mean(sum(abs2, ground_truth; dims = 1))),
    )
end

function get_valid_time(
    ground_truth::AbstractMatrix{T},
    predicted::AbstractMatrix{T},
    times::AbstractVector{T};
    error_threshold::T,
) where {T<:AbstractFloat}
    normalised_error = get_normalised_error(ground_truth, predicted)
    return get_valid_time(normalised_error, times; error_threshold)
end

function get_valid_time(
    normalised_error::AbstractVector{T},
    times::AbstractVector{T};
    error_threshold::T,
) where {T<:AbstractFloat}
    error_threshold_index = findfirst(error -> error >= error_threshold, normalised_error)
    if isnothing(error_threshold_index)
        valid_time = times[end]
    else
        valid_time = times[error_threshold_index - 1]
    end
    return valid_time
end
