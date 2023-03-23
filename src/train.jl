function train!(
    θ::AbstractVector{T},
    prob::SciMLBase.AbstractDEProblem,
    data::Data{T},
    epochs::Int,
    optimiser::Optimisers.AbstractRule,
    scheduler::ParameterSchedulers.AbstractSchedule;
    loss::Function = MSE,
    solver::SciMLBase.AbstractDEAlgorithm = Tsit5(),
    adjoint::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm = BacksolveAdjoint(;
        autojacvec = ReverseDiffVJP(true),
    ),
    reltol::T = 1.0f-6,
    abstol::T = 1.0f-6,
    patience = Inf,
    time_limit = 23 * 60 * 60.0f0,
    verbose = false,
    show_plot = false,
) where {T<:AbstractFloat}
    @info "Beginning training..."

    (; train_data, val_data, test_data) = data

    # Keep track of the minimum validation loss and parameters for early stopping
    θ_min = copy(θ)
    min_val_loss = Inf32
    min_val_epoch = 0
    early_stopping = Flux.early_stopping(loss -> loss, patience; init_score = min_val_loss)

    # Keep track of training loss, validation loss, and duration per epoch
    learning_curve = Array{Array{Float32}}(undef, 0)

    opt_state = Optimisers.setup(optimiser, θ)

    training_start_time = time()
    for (epoch, learning_rate) in zip(1:epochs, scheduler)
        Optimisers.adjust!(opt_state, learning_rate)

        iter = 0
        training_losses = Float32[]
        epoch_start_time = time()
        for (times, target_trajectory) in shuffle(train_data)
            iter += 1
            tspan = (times[1], times[end])
            u0 = target_trajectory[:, 1]
            prob = remake(prob; u0, tspan)

            training_loss, gradients = Zygote.withgradient(θ) do θ
                predicted_trajectory = predict(
                    prob,
                    θ;
                    solver,
                    saveat = times,
                    reltol,
                    abstol,
                    sensealg = adjoint,
                )
                return loss(predicted_trajectory, target_trajectory)
            end

            push!(training_losses, training_loss)
            opt_state, θ = Optimisers.update!(opt_state, θ, gradients[1])

            if verbose
                @info @sprintf "[epoch = %04i] [iter = %04i] [tspan = (%05.2f, %05.2f)] Loss = %.2e\n" epoch iter tspan[1] tspan[2] training_loss
            end
        end
        val_loss = evaluate(prob, θ, val_data, loss, solver, reltol, abstol)
        epoch_duration = time() - epoch_start_time

        @info @sprintf "[epoch = %04i] Learning rate = %.1e" epoch learning_rate
        @info @sprintf "[epoch = %04i] Train loss = %.2e\n" epoch mean(training_losses)
        @info @sprintf "[epoch = %04i] Valid loss = %.2e\n" epoch val_loss
        @info @sprintf "[epoch = %04i] Duration = %.1f seconds\n" epoch epoch_duration

        push!(
            learning_curve,
            [epoch, learning_rate, mean(training_losses), val_loss, epoch_duration],
        )

        early_stopping(val_loss) && break

        if val_loss < min_val_loss
            θ_min = copy(θ)
            min_val_epoch = epoch
            min_val_loss = val_loss
        end

        if (time() - training_start_time) > time_limit
            @info @sprintf "Time limit of %.1f hours reached." time_limit / 3600
            break
        end

        flush(stderr)  # So we can watch log files on the cluster
    end
    training_duration = time() - training_start_time

    # Evaluate trained model
    θ .= θ_min
    test_loss = evaluate(prob, θ, test_data, loss, solver, reltol, abstol)

    @info "Training complete."
    @info @sprintf "Minimum validation loss = %.2e\n" min_val_loss
    @info @sprintf "Test loss = %.2e\n" test_loss
    @info @sprintf "Training duration = %.1f seconds\n" training_duration

    return training_duration, learning_curve, min_val_epoch, min_val_loss, test_loss
end
