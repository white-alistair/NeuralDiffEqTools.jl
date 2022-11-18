function get_prob(rhs, θ::Vector{T}, N) where {T}
    return ODEProblem(rhs, rand(T, N), (T(0), T(1)), θ)
end
