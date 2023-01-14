struct Lesson{T}
    steps::Int
    epochs::Int
    optimiser::AbstractOptimiser{T}
end
