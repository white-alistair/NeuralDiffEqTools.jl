struct MultipleShooting{T}
    time_series::TimeSeries{T}
    step_size::Int
    shuffle::Bool
end

MultipleShooting(time_series::TimeSeries{T}, step_size::Int) where {T} =
    MultipleShooting{T}(time_series, step_size, true)

Base.length(ms::MultipleShooting) = floor(Int32, length(ms.time_series) / ms.step_size)
Base.eltype(::Type{MultipleShooting{T}}) where {T} = Tuple{Vector{T},Matrix{T}}

function Base.iterate(ms::MultipleShooting)
    N = length(ms.time_series)
    start_indexes = collect(1:ms.step_size:N-ms.step_size)
    if ms.shuffle
        shuffle!(start_indexes)
    end
    iterate(ms, start_indexes)
end

function Base.iterate(ms::MultipleShooting, start_indexes)
    isempty(start_indexes) && return nothing
    start_index = pop!(start_indexes)
    times = @view ms.time_series.times[start_index:start_index+ms.step_size]
    trajectory = @view ms.time_series.trajectory[:, start_index:start_index+ms.step_size]
    return ((times, trajectory), start_indexes)
end
