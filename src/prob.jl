function get_prob(rhs, θ::Vector{T}, N; in_place = true, specialize = SciMLBase.FullSpecialize) where {T}
    return ODEProblem{in_place,specialize}(rhs, rand(T, N), (T(0), T(1)), θ)
end
