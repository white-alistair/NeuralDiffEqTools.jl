function multiple_shooting(time_series::TimeSeries, step::Int, shuffle = true)
    (; times, trajectory) = time_series
    start_indexes = collect(1:step:length(time_series)-step)
    if shuffle
        shuffle!(start_indexes)
    end
    return [(times[i:i+step], trajectory[:, i:i+step]) for i in start_indexes]
end
