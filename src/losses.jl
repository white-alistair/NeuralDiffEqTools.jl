function MSE(prediction::AbstractMatrix{T}, target::AbstractMatrix{T}) where {T<:Real}
    return mean(abs2, prediction .- target)
end

L1(θ) = sum(abs, θ)
L2(θ) = sum(abs2, θ)
