function cb_display(epoch, steps, iter, tspan, loss, target, pred, t; verbose, show_plot)
    if verbose
        @info @sprintf "[epoch = %04i] [steps = %02i] [iter = %04i] [tspan = (%05.2f, %05.2f)] Loss = %.2e\n" epoch steps iter tspan[1] tspan[2] loss
    end
    # if show_plot
    #     plot_prediction(system, target, pred, t)  # Method defined in the same file as the given system
    # end
    return false
end

function plot_prediction(target, pred, t, layout, title)
    plt = Plots.plot(t, transpose(target), layout = layout, title = title, label = "target")
    Plots.plot!(plt, t, transpose(pred), layout = layout, label = "predicted")
    display(plt)
end
