struct LearningCurve{T}
    epoch::Vector{Int}
    learning_rate::Vector{T}
    train_loss::Vector{T}
    val_loss::Vector{T}
    duration::Vector{T}
end

function LearningCurve{T}() where {T}
    return LearningCurve{T}([], [], [], [], [])
end

import Base: push!
function push!(lc::LearningCurve, epoch, lr, train_loss, val_loss, duration)
    push!(lc.epoch, epoch)
    push!(lc.learning_rate, lr)
    push!(lc.train_loss, train_loss)
    push!(lc.val_loss, val_loss)
    push!(lc.duration, duration)
    return nothing
end

function save_learning_curve(lc::LearningCurve, model_id; dir = "learning_curves")
    col_names = ["epoch" "learning_rate" "train_loss" "val_loss" "duration"]
    cols = [lc.epoch lc.learning_rate lc.train_loss lc.val_loss lc.duration]
    mkpath(dir)
    path = joinpath(dir, model_id * ".csv")
    open(path, "w") do io
        writedlm(io, col_names, ',')
        writedlm(io, cols, ',')
    end
    return nothing
end

struct IntegerTicks end
Makie.get_tickvalues(::IntegerTicks, vmin, vmax) = ceil(Int, vmin):floor(Int, vmax)

function plot_learning_curve(model_id::Integer)
    filepath = "learning_curves/$(model_id).csv"
    return plot_learning_curve(filepath)
end

function plot_learning_curve(lc::LearningCurve)
    (; epoch, learning_rate, train_loss, val_loss, duration) = lc
    return plot_learning_curve(
        epoch,
        learning_rate,
        train_loss,
        val_loss,
        duration,
    )
end

function plot_learning_curve(filepath::String)
    data, header = readdlm(filepath, ','; header = true)
    epoch, learning_rate, training_loss, validation_loss, duration = eachcol(data)
    return plot_learning_curve(
        epoch,
        learning_rate,
        training_loss,
        validation_loss,
        duration,
    )
end

function plot_learning_curve(epoch, learning_rate, training_loss, validation_loss, duration)
    f = CairoMakie.Figure(; resolution = (1000, 1000))

    # 1. Plot training and validation loss
    ax1 = Axis(
        f[1:4, 1];
        ylabel = "validation loss",
        yscale = log10,
        yticks = LogTicks(IntegerTicks()),
        yaxisposition = :right,
        ygridvisible = false,
    )

    ax2 = Axis(
        f[1:3, 1];
        ylabel = "training loss",
        yscale = log10,
        yticks = LogTicks(IntegerTicks()),
        ygridvisible = true,
        yminorticksvisible = true,
        yminorgridvisible = true,
        yminorticks = IntervalsBetween(8),
    )
    hidespines!(ax2)
    hidexdecorations!(ax2)

    train_plot = lines!(ax2, epoch, training_loss; color = :blue)
    valid_plot = lines!(ax1, epoch, validation_loss; color = :red)

    min_val_loss, min_val_epoch = findmin(validation_loss)
    min_val_plot = hlines!(ax1, min_val_loss; linestyle = :dash)
    vlines!(ax1, min_val_epoch; linestyle = :dash)
    Legend(
        f[1:4, 1],
        [train_plot, valid_plot, min_val_plot],
        ["training loss", "validation loss", "minimum validation loss"];
        margin = (10, 10, 10, 10),
        tellheight = false,
        tellwidth = false,
        halign = :right,
        valign = :top,
    )

    # 2. Plot learning rate
    ax3 = Axis(
        f[5, 1];
        ylabel = "learning rate",
        ytickformat = (labels -> [@sprintf "%.e" l for l in labels]),
    )
    linkxaxes!(ax1, ax3)
    lines!(ax3, epoch, learning_rate)

    # 3. Plot epoch duration
    ax4 = Axis(f[7, 1]; xlabel = "epoch", ylabel = "duration [s]")
    linkxaxes!(ax1, ax4)
    lines!(ax4, epoch, duration)

    return f
end
