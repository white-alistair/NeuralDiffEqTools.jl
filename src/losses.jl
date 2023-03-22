function MSE(prediction::AbstractMatrix{T}, target::AbstractMatrix{T}) where {T<:Real}
    return mean(abs2, prediction[:, 2:end] .- target[:, 2:end])  # Do not include u0
end
