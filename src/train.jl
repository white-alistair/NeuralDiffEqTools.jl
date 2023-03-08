function train!(
    θ::Vector{T},
    prob::SciMLBase.AbstractDEProblem,
    data::KLFolds{T},
    curriculum::Curriculum{T},
    loss::Function,
    solver::SciMLBase.AbstractDEAlgorithm = Tsit5(),
    adjoint::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm = BacksolveAdjoint(;
        autojacvec = ReverseDiffVJP(true),
    );
    reltol::T = 1.0f-6,
    abstol::T = 1.0f-6,
    maxiters = 10_000,
    valid_error_threshold::T = 4.0f-1,
    stopping_criterion::Symbol,  # :val_loss or :valid_time
    patience = Inf,
    time_limit = 23 * 60 * 60.0f0,
    verbose = false,
    show_plot = false,
) where {T<:AbstractFloat}
    (; k, l) = data
    kl_period = Int(k / l)  # Number of epochs for one kl-fold cycle, i.e. each fold is used once for validation

    @info "Beginning training..."
    @info @sprintf "Applying kl-fold cross validation with k = %i, l = %i" k l

    # Keep track of the minimum validation loss and parameters for early stopping
    θ_min = copy(θ)
    early_stopping_val_loss = Inf32
    early_stopping_valid_time = zero(T)
    early_stopping_epoch = 0
    # early_stopping = Flux.early_stopping(loss -> loss, patience; init_score = early_stopping_val_loss)

    # Keep track of training loss, validation loss, and duration per epoch
    learning_curve = Array{Array{Float32}}(undef, 0)

    epoch = 0
    val_losses = Float32[]
    valid_times = Float32[]
    training_start_time = time()
    for lesson in curriculum.lessons
        lesson_start_time = time()
        (; name, steps, epochs, optimiser) = lesson
        set_initial_learning_rate!(optimiser)

        for (training_folds, validation_folds) in kl_cycle(data, epochs)
            epoch += 1
            iter = 0
            training_losses = Float32[]

            #! format: off
            @info @sprintf "[lesson = %-20.20s] [epoch = %04i] Learning rate = %.1e" name epoch get_learning_rate(optimiser)
            #! format: on

            epoch_start_time = time()
            for fold in training_folds
                for (times, target_trajectory) in MultipleShooting(fold, steps)
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
                        return loss(predicted_trajectory, target_trajectory, θ)
                    end

                    Optimisers.update!(optimiser, θ, gradients)

                    push!(training_losses, training_loss)

                    if verbose
                        @info @sprintf "[lesson = %-20.20s] [epoch = %04i] [iter = %04i] [tspan = (%05.2f, %05.2f)] Loss = %.2e\n" name epoch iter tspan[1] tspan[2] training_loss
                    end
                end
            end

            val_loss, valid_time = evaluate(
                prob,
                θ,
                validation_folds,
                loss,
                solver,
                reltol,
                abstol;
                valid_error_threshold,
                maxiters,
                show_plot,
            )
            push!(val_losses, val_loss)
            push!(valid_times, valid_time)

            # Compute the moving average over the last kl_period epochs
            moving_avg_index = max(1, length(val_losses) - kl_period + 1)
            moving_avg_val_loss = mean(val_losses[moving_avg_index:end])
            moving_avg_valid_time = mean(valid_times[moving_avg_index:end])

            epoch_duration = time() - epoch_start_time

            #! format: off
            @info @sprintf "[lesson = %-20.20s] [epoch = %04i] Average training loss = %.2e\n" name epoch mean(training_losses)
            @info @sprintf "[lesson = %-20.20s] [epoch = %04i] Moving average validation loss = %.2e\n" name epoch moving_avg_val_loss
            @info @sprintf "[lesson = %-20.20s] [epoch = %04i] Moving average valid time = %.1f seconds\n" name epoch moving_avg_valid_time
            @info @sprintf "[lesson = %-20.20s] [epoch = %04i] Epoch duration = %.1f seconds\n" name epoch epoch_duration
            #! format: on

            push!(
                learning_curve,
                [
                    epoch,
                    steps,
                    get_learning_rate(optimiser),
                    mean(training_losses),
                    moving_avg_val_loss,
                    moving_avg_valid_time,
                    epoch_duration,
                ],
            )

            # early_stopping(val_loss) && @goto complete_training  # Use goto and label to break out of nested loops
            if (
                stopping_criterion == :val_loss &&
                moving_avg_val_loss < early_stopping_val_loss
            ) || (
                stopping_criterion == :valid_time &&
                moving_avg_valid_time > early_stopping_valid_time
            )
                θ_min = copy(θ)
                early_stopping_epoch = epoch
                early_stopping_val_loss = moving_avg_val_loss
                early_stopping_valid_time = moving_avg_valid_time
            end

            if (time() - training_start_time) > time_limit
                 #! format: off
                 @info @sprintf "[lesson = %-20.20s] [epoch = %04i] Time limit of %.1f hours reached for the training loop. Stopping here." name epoch (time_limit / 3600)
                 @goto complete_training  # Use goto and label to break out of nested loops
                 #! format: on
            end

            if optimiser isa AbstractScheduledOptimiser
                update_learning_rate!(optimiser)
            end
        end

        flush(stderr)  # Keep log files up to date on the cluster
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
        data.test_folds,
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

function train!(
    θ::Vector{T},
    prob::SciMLBase.AbstractDEProblem,
    data::TrainValidTestSplit{T},
    curriculum::Curriculum{T},
    loss::Function,
    solver::SciMLBase.AbstractDEAlgorithm = Tsit5(),
    adjoint::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm = BacksolveAdjoint(;
        autojacvec = ReverseDiffVJP(true),
    );
    reltol::T = 1.0f-6,
    abstol::T = 1.0f-6,
    maxiters = 10_000,
    valid_error_threshold::T = 4.0f-1,
    stopping_criterion::Symbol,  # :val_loss or :valid_time
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

    epoch = 0
    training_start_time = time()
    for lesson in curriculum.lessons
        (; name, steps, epochs, optimiser) = lesson
        set_initial_learning_rate!(optimiser)

        lesson_start_time = time()
        for _ = 1:epochs
            epoch += 1
            training_losses = Float32[]

            #! format: off
            @info @sprintf "[lesson = %-20.20s] [epoch = %04i] Learning rate = %.1e" name epoch get_learning_rate(optimiser)
            #! format: on

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
                    return loss(predicted_trajectory, target_trajectory, θ)
                end

                Optimisers.update!(optimiser, θ, gradients)

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
                    get_learning_rate(optimiser),
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

            if optimiser isa AbstractScheduledOptimiser
                update_learning_rate!(optimiser)
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

function train!(
    θ::Vector{T},
    prob::SciMLBase.AbstractDEProblem,
    data::AbstractData{T},
    curriculum::Curriculum{T};
    # Solver
    solver::SciMLBase.AbstractDEAlgorithm = Tsit5(),
    reltol::T = 1.0f-6,
    abstol::T = 1.0f-6,
    maxiters = 10_000,
    # Adjoint
    sensealg::Union{Symbol,Nothing} = :BacksolveAdjoint,
    vjp::Union{Symbol,Nothing} = :ReverseDiffVJP,
    checkpointing::Bool = false,
    # Regularisation
    penalty = L2,
    regularisation_param::T = 0.0f0,
    # Validation and early Stopping
    valid_error_threshold::T = 4.0f-1,
    stopping_criterion::Symbol,  # :val_loss or :valid_time
    patience = Inf,
    time_limit = 23 * 60 * 60.0f0,
    # I/O
    verbose = false,
    show_plot = false,
) where {T<:AbstractFloat}
    # 1. Set up the loss function
    loss = (pred, target, θ) -> MSE(pred, target) + regularisation_param * penalty(θ)

    # 2. Set up the adjoint sensitivity algorithm for computing gradients of the ODE solve
    adjoint = get_adjoint(sensealg, vjp, checkpointing)

    return train!(
        θ,
        prob,
        data,
        curriculum,
        loss,
        solver,
        adjoint;
        reltol,
        abstol,
        maxiters,
        valid_error_threshold,
        stopping_criterion,
        patience,
        time_limit,
        verbose,
        show_plot,
    )
end
