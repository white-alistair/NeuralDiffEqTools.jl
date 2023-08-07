function multiple_shooting(time_series::TimeSeries; steps::Int, shuffle = false)
    (; times, trajectory) = time_series
    start_indexes = collect(1:steps:length(time_series)-steps)
    if shuffle
        shuffle!(start_indexes)
    end
    return [(times[i:i+steps], trajectory[:, i:i+steps]) for i in start_indexes]
end
