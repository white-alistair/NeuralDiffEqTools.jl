function cb_display(epoch, steps, iter, tspan, loss, ground_truth, prediction, times; verbose, show_plot)
    if verbose
        @info @sprintf "[epoch = %04i] [iter = %04i] [steps = %02i] [tspan = (%05.2f, %05.2f)] Loss = %.2e\n" epoch steps iter tspan[1] tspan[2] loss
    end
    if show_plot
        plot_prediction(ground_truth, prediction, times)
    end
    return false
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
