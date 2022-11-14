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
    shuffle::Bool
end
DataLoader(data::TimeSeries{T}, step_size::Int) where T = DataLoader{T}(data, step_size, true)

Base.length(dl::DataLoader) = floor(Int32, length(dl.data) / dl.step_size)
Base.eltype(::Type{DataLoader{T}}) where {T} = Tuple{Vector{T}, Matrix{T}}

function Base.iterate(dl::DataLoader)
    N = length(dl.data)
    start_indexes = collect(1:dl.step_size:N-dl.step_size)
    if dl.shuffle
        shuffle!(start_indexes)
    end
    iterate(dl, start_indexes)
end

function Base.iterate(dl::DataLoader, start_indexes)
    isempty(start_indexes) && return nothing
    start_index = pop!(start_indexes)
    times = dl.data.times[start_index:start_index + dl.step_size]
    trajectory = dl.data.trajectory[:, start_index:start_index + dl.step_size]
    return ((times, trajectory), start_indexes)
end
