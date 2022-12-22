struct TimeSeries{T<:AbstractFloat}
    trajectory::Matrix{T}
    times::Vector{T}
    TimeSeries{T}(trajectory, times) where {T<:AbstractFloat} =
        size(trajectory)[end] != size(times)[end] ?
        throw(DimensionMismatch("number of times and observations do not match")) :
        new(trajectory, times)
end

function TimeSeries(trajectory::Matrix{T}, times::Vector{T}) where {T<:AbstractFloat}
    TimeSeries{T}(trajectory, times)
end

function TimeSeries{T}(sol::OrdinaryDiffEq.ODESolution) where {T<:AbstractFloat}
    return TimeSeries{T}(Array(sol), sol.t)
end

Base.getindex(ode_solution::TimeSeries, i::Int) = ode_solution.trajectory[:, i]
Base.getindex(ode_solution::TimeSeries, I::UnitRange) = ode_solution.trajectory[:, I]
Base.firstindex(ode_solution::TimeSeries) = 1
Base.lastindex(ode_solution::TimeSeries) = size(ode_solution.trajectory)[2]
Base.length(ode_solution::TimeSeries) = length(ode_solution.times)

struct Data{T<:AbstractFloat}
    train_data::TimeSeries{T}
    val_data::Array{TimeSeries{T}}
    test_data::Array{TimeSeries{T}}
end

# DATA LOADER
struct DataLoader{T<:AbstractFloat}
    data::TimeSeries{T}
    step_size::Int
    batch_size::Int
    shuffle::Bool
end
DataLoader(data::TimeSeries{T}, step_size::Int, batch_size::Int) where {T} =
    DataLoader{T}(data, step_size, batch_size, true)

Base.length(dl::DataLoader) = floor(Int32, length(dl.data) / dl.step_size)
Base.eltype(::Type{DataLoader{T}}) where {T} = Tuple{Vector{T},Matrix{T}}

function Base.iterate(dl::DataLoader)
    N = length(dl.data)

    # 1. For a given step size, find the start index of each step
    start_indexes = collect(1:dl.step_size:N-dl.step_size)
    if dl.shuffle
        shuffle!(start_indexes)
    end

    # 2. Batch the step indexes according to the given batch size
    batch_index_iterator = Iterators.partition(start_indexes, dl.batch_size)

    return iterate(dl, (batch_index_iterator, 1))
end

function Base.iterate(dl::DataLoader, (batch_index_iterator, state))
    next = iterate(batch_index_iterator, state)
    isnothing(next) && return nothing
    (batch_indexes, state) = next

    batch_times = [@view dl.data.times[i:i+dl.step_size] for i in batch_indexes]
    batch_trajectories = [@view dl.data.trajectory[:, i:i+dl.step_size] for i in batch_indexes]

    return ((batch_times, batch_trajectories), (batch_index_iterator, state))
end
