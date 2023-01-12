function evaluate(
    prob,
    θ,
    ground_truth::TimeSeries{T},
    loss_function,
    solver,
    reltol,
    abstol;
    maxiters,
    error_threshold::T = 0.4f0,
    show_plot,
) where {T<:AbstractFloat}
    times = ground_truth.times
    target_trajectory = ground_truth.trajectory
    u0 = target_trajectory[:, 1]
    tspan = (times[1], times[end])
    prob = remake(prob; u0, tspan)

    retcode, predicted_trajectory = predict(prob, θ; solver, saveat = times, reltol, abstol, maxiters)
    if retcode != :Success
        @warn "prediction failed, skipping"
        return T(NaN), T(NaN)
    end

    loss = loss_function(predicted_trajectory, target_trajectory, θ)
    valid_time =
        get_valid_time(target_trajectory, predicted_trajectory, times; error_threshold)

    if show_plot
        plot_prediction(target_trajectory, predicted_trajectory, ground_truth.times)
        sleep(0.0)
    end

    return loss, valid_time
end

function evaluate(
    prob,
    θ,
    ground_truth::AbstractVector{TimeSeries{T}},
    loss_function,
    solver,
    reltol,
    abstol;
    maxiters,
    error_threshold::T = 0.4f0,
    show_plot,
) where {T<:AbstractFloat}
    losses = T[]
    valid_times = T[]

    for reference in ground_truth
        loss, valid_time = evaluate(
            prob,
            θ,
            reference,
            loss_function,
            solver,
            reltol,
            abstol;
            maxiters,
            error_threshold,
            show_plot,
        )
        push!(losses, loss)
        push!(valid_times, valid_time)
        GC.gc(false)
    end

    return mean(skipmissing(losses)), mean(skipmissing(valid_times))
end

function plot_prediction(ground_truth, predicted, times)
    fig = CairoMakie.Figure()
    dim = size(ground_truth)[1]
    for i in 1:dim
        ax = Axis(fig[i, 1])
        lines_ground_truth = lines!(ax, times, ground_truth[i, :])
        lines_predicted = lines!(ax, times, predicted[i, :])
        if i == dim
            Legend(fig[1:dim, 2], [lines_ground_truth, lines_predicted], ["ground truth", "predicted"])
        end
    end
    display(fig)
end
