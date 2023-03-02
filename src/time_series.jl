struct TimeSeries{T}
    times::Vector{T}
    trajectory::Matrix{T}
    TimeSeries{T}(times, trajectory) where {T} =
        size(times)[end] != size(trajectory)[end] ?
        throw(DimensionMismatch("number of times and observations do not match")) :
        new(times, trajectory)
end

function TimeSeries(times::Vector{T}, trajectory::Matrix{T}) where {T}
    TimeSeries{T}(times, trajectory)
end

function TimeSeries{T}(sol::OrdinaryDiffEq.ODESolution) where {T}
    return TimeSeries{T}(sol.t, Array(sol))
end

Base.getindex(ode_solution::TimeSeries, i::Int) = ode_solution.trajectory[:, i]
Base.getindex(ode_solution::TimeSeries, I::UnitRange) = ode_solution.trajectory[:, I]
Base.firstindex(ode_solution::TimeSeries) = 1
Base.lastindex(ode_solution::TimeSeries) = size(ode_solution.trajectory)[2]
Base.length(ode_solution::TimeSeries) = length(ode_solution.times)

function chunk(time_series::TimeSeries{T}, chunk_size::Int) where {T}
    (; times, trajectory) = time_series
    last_dim = ndims(trajectory)
    N = size(trajectory)[end]
    
    chunks = TimeSeries{T}[]
    for i in 1:(chunk_size-1):N
        time_chunk = times[i:min(i + chunk_size - 1, N)]
        traj_chunk = selectdim(trajectory, last_dim, i:min(i + chunk_size - 1, N)) 
        push!(chunks, TimeSeries{T}(time_chunk, traj_chunk))
    end

    return chunks
end
