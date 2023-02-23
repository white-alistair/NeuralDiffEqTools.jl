function MSE(prediction::AbstractMatrix{T}, target::AbstractMatrix{T}) where {T<:Real}
    return mean(abs2, prediction .- target)
end

L1(θ) = norm(θ, 1)
L2(θ) = norm(θ, 2)
