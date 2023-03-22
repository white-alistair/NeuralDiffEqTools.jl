function train!(
    θ::Vector{T},
    prob::SciMLBase.AbstractDEProblem,
    data::TrainValidTestSplit{T},
    curriculum::Curriculum;
    loss::Function = MSE,
    solver::SciMLBase.AbstractDEAlgorithm = Tsit5(),
    adjoint::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm = BacksolveAdjoint(;
        autojacvec = ReverseDiffVJP(true),
    ),
    reltol::T = 1.0f-6,
    abstol::T = 1.0f-6,
    maxiters = 10_000,
    valid_error_threshold::T = 4.0f-1,
    stopping_criterion::Symbol = :valid_time,  # :val_loss or :valid_time
    patience = Inf,
    time_limit = 23 * 60 * 60.0f0,
    verbose = false,
    show_plot = false,
) where {T<:AbstractFloat}
    @info "Beginning training..."

    (; train_data, valid_data, test_data) = data

    # Keep track of the minimum validation loss and parameters for early stopping
    θ_min = copy(θ)
    early_stopping_val_loss = Inf32
    early_stopping_valid_time = zero(T)
    early_stopping_epoch = 0
    # early_stopping = Flux.early_stopping(loss -> loss, patience; init_score = early_stopping_val_loss)

    # Keep track of training loss, validation loss, and duration per epoch
    learning_curve = Array{Array{Float32}}(undef, 0)

    local opt_state
    
    epoch = 0
    training_start_time = time()
    for lesson in curriculum.lessons
        (; name, steps, epochs, optimiser, scheduler) = lesson
        
        # If no optimiser is given, we re-use the one from the previous lesson
        if !isnothing(optimiser)
            opt_state = Optimisers.setup(optimiser, θ)
        end

        lesson_start_time = time()
        for _ = 1:epochs
            epoch += 1
            training_losses = Float32[]

            learning_rate = ParameterSchedulers.next!(scheduler)
            Optimisers.adjust!(opt_state; eta = learning_rate)
            @info @sprintf "[lesson = %-20.20s] [epoch = %04i] Learning rate = %.1e" name epoch learning_rate

            iter = 0
            epoch_start_time = time()
            for (times, target_trajectory) in MultipleShooting(train_data, steps)
                iter += 1
                tspan = (times[1], times[end])
                u0 = target_trajectory[:, 1]
                prob = remake(prob; u0, tspan)

                training_loss, gradients = Zygote.withgradient(θ) do θ
                    retcode, predicted_trajectory = predict(
                        prob,
                        θ;
                        solver,
                        saveat = times,
                        reltol,
                        abstol,
                        maxiters,
                        sensealg = adjoint,
                    )
                    return loss(predicted_trajectory, target_trajectory)
                end

                opt_state, θ = Optimisers.update!(opt_state, θ, gradients[1])

                push!(training_losses, training_loss)

                if verbose
                    @info @sprintf "[lesson = %-20.20s] [epoch = %04i] [iter = %04i] [tspan = (%05.2f, %05.2f)] Loss = %.2e\n" name epoch iter tspan[1] tspan[2] training_loss
                end
            end
            epoch_duration = time() - epoch_start_time

            val_loss, valid_time = evaluate(
                prob,
                θ,
                valid_data,
                loss,
                solver,
                reltol,
                abstol;
                valid_error_threshold,
                maxiters,
                show_plot,
            )

            #! format: off
            @info @sprintf "[lesson = %-20.20s] [epoch = %04i] Average training loss = %.2e\n" name epoch mean(training_losses)
            @info @sprintf "[lesson = %-20.20s] [epoch = %04i] Validation loss = %.2e\n" name epoch val_loss
            @info @sprintf "[lesson = %-20.20s] [epoch = %04i] Valid time = %.1f seconds\n" name epoch valid_time
            @info @sprintf "[lesson = %-20.20s] [epoch = %04i] Epoch duration = %.1f seconds\n" name epoch epoch_duration
            #! format: on

            push!(
                learning_curve,
                [
                    epoch,
                    steps,
                    learning_rate,
                    mean(training_losses),
                    val_loss,
                    valid_time,
                    epoch_duration,
                ],
            )

            # early_stopping(val_loss) && @goto complete_training  # Use goto and label to break out of nested loops
            if (stopping_criterion == :val_loss && val_loss < early_stopping_val_loss) ||
               (stopping_criterion == :valid_time && valid_time > early_stopping_valid_time)
                θ_min = copy(θ)
                early_stopping_epoch = epoch
                early_stopping_val_loss = val_loss
                early_stopping_valid_time = valid_time
            end

            if (time() - training_start_time) > time_limit
                #! format: off
                @info @sprintf "[lesson = %-20.20s] [epoch = %04i] Time limit of %.1f hours reached for the training loop. Stopping here." name epoch (time_limit / 3600)
                @goto complete_training  # Use goto and label to break out of nested loops
                #! format: on
            end
            flush(stderr)  # Keep log files up to date on the cluster
        end
        lesson_duration = time() - lesson_start_time
        @info @sprintf "[lesson = %-20.20s] Lesson duration = %.1f seconds\n" name lesson_duration
    end

    @label complete_training
    training_duration = time() - training_start_time

    # Evaluate trained model
    θ .= θ_min
    test_loss, test_valid_time = evaluate(
        prob,
        θ,
        test_data,
        loss,
        solver,
        reltol,
        abstol;
        valid_error_threshold,
        maxiters,
        show_plot,
    )

    @info "Training complete."
    @info @sprintf "Early stopping validation loss = %.2e\n" early_stopping_val_loss
    @info @sprintf "Early stopping valid time = %.1f seconds\n" early_stopping_valid_time
    @info @sprintf "Test loss = %.2e\n" test_loss
    @info @sprintf "Test valid time = %.1f seconds\n" test_valid_time
    @info @sprintf "Training duration = %.1f seconds\n" training_duration

    return training_duration,
    learning_curve,
    epoch,
    early_stopping_epoch,
    early_stopping_val_loss,
    early_stopping_valid_time,
    test_loss,
    test_valid_time
end
