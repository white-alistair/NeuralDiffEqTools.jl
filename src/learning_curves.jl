function save_learning_curve(learning_curve, model_id; dir = "learning_curves")
    col_names = [["epoch", "steps", "learning_rate", "train_loss", "val_loss", "duration"]]
    mkpath(dir)
    path = joinpath(dir, model_id * ".csv")
    open(path, "w") do io
        writedlm(io, col_names, ',')
        writedlm(io, learning_curve, ',')
    end
    return nothing
end

struct IntegerTicks end
Makie.get_tickvalues(::IntegerTicks, vmin, vmax) = ceil(Int, vmin):floor(Int, vmax)

function plot_learning_curve(model_id::Integer)
    filepath = "learning_curves/$(model_id).csv"
    return plot_learning_curve(filepath)
end

function plot_learning_curve(filepath::String)
    data, header = readdlm(filepath, ',', header = true)
    epoch, steps, learning_rate, training_loss, validation_loss, duration = eachcol(data)
    return plot_learning_curve(
        epoch,
        steps,
        learning_rate,
        training_loss,
        validation_loss,
        duration,
    )
end

function plot_learning_curve(
    epoch,
    steps,
    learning_rate,
    training_loss,
    validation_loss,
    duration,
)
    f = CairoMakie.Figure(resolution = (1000, 1000))

    step_epochs = [epoch[findfirst(step .== steps)] for step in unique(steps)]
    step_labels = string.(Int.(unique(steps)))

    # 1. Plot training and validation loss
    ax1 = Axis(
        f[1:3, 1],
        ylabel = "validation loss",
        yscale = log10,
        yaxisposition = :right,
        xticks = step_epochs,
        yminorticksvisible = true,
        yminorgridvisible = true,
        yminorticks = IntervalsBetween(8),
    )

    ax2 = Axis(f[1:3, 1], ylabel = "training loss", yscale = log10, ygridvisible = false)
    hidespines!(ax2)
    hidexdecorations!(ax2)

    train_plot = lines!(ax2, epoch, training_loss, color = :blue)
    valid_plot = lines!(ax1, epoch, validation_loss, color = :red)

    min_val_loss, min_val_epoch = findmin(validation_loss)
    min_val_plot = hlines!(ax1, min_val_loss, linestyle = :dash)
    vlines!(ax1, min_val_epoch, linestyle = :dash)
    Legend(
        f[1:3, 1],
        [train_plot, valid_plot, min_val_plot],
        ["training loss", "validation loss", "minimum validation loss"],
        margin = (10, 10, 10, 10),
        tellheight = false,
        tellwidth = false,
        halign = :right,
        valign = :top,
    )

    # 2. Plot learning rate
    ax3 = Axis(
        f[4, 1],
        xticks = step_epochs,
        yscale = log10,
        yticks = LogTicks(IntegerTicks()),
        ylabel = "learning rate",
    )
    linkxaxes!(ax1, ax3)
    lines!(ax3, epoch, learning_rate)

    # 3. Plot epoch duration
    ax4 = Axis(
        f[5, 1],
        xticks = step_epochs,
        xlabel = "epoch",
        ylabel = "duration",
    )
    linkxaxes!(ax1, ax4)
    lines!(ax4, epoch, duration)

    # 4. Add labels and ticks for integration steps
    ax5 = Axis(
        f[1:3, 1],
        xlabel = "integration steps",
        xaxisposition = :top,
        xticks = (step_epochs, step_labels),
        xticklabelsvisible = true,
    )
    hidespines!(ax5)
    hideydecorations!(ax5)
    linkxaxes!(ax1, ax5)

    ax6 = Axis(
        f[4, 1],
        xaxisposition = :top,
        xticks = (step_epochs, step_labels),
        xticklabelsvisible = true,
    )
    hidespines!(ax6)
    hideydecorations!(ax6)
    linkxaxes!(ax1, ax6)

    ax7 = Axis(
        f[5, 1],
        xaxisposition = :top,
        xticks = (step_epochs, step_labels),
        xticklabelsvisible = true,
    )
    hidespines!(ax7)
    hideydecorations!(ax7)
    linkxaxes!(ax1, ax7)

    return f
end
