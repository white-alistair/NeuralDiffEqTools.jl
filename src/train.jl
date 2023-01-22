function train!(
    θ::Vector{T},
    prob::SciMLBase.AbstractDEProblem,
    data::Data{T},
    curriculum::Curriculum{T},
    loss::Function,
    solver::SciMLBase.AbstractDEAlgorithm = Tsit5(),
    adjoint::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm = BacksolveAdjoint(;
        autojacvec = ReverseDiffVJP(true),
    );
    reltol::T = 1.0f-6,
    abstol::T = 1.0f-6,
    maxiters = 10_000,
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
    min_val_valid_time = NaN32
    early_stopping = Flux.early_stopping(loss -> loss, patience; init_score = min_val_loss)

    # Keep track of training loss, validation loss, and duration per epoch
    learning_curve = Array{Array{Float32}}(undef, 0)

    epoch = 0
    training_start_time = time()
    for lesson in curriculum.lessons
        (; name, steps, epochs, optimiser) = lesson
        set_initial_learning_rate!(optimiser)

        data_loader = DataLoader(train_data, steps)

        lesson_start_time = time()
        for _ = 1:epochs
            epoch += 1
            training_losses = Float32[]

            #! format: off
            @info @sprintf "[lesson = %-20.20s] [epoch = %04i] Learning rate = %.1e" name epoch get_learning_rate(optimiser)
            #! format: on

            iter = 0
            epoch_start_time = time()
            for (times, target_trajectory) in data_loader
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

            val_loss, val_valid_time = evaluate(
                prob,
                θ,
                val_data,
                loss,
                solver,
                reltol,
                abstol;
                maxiters,
                show_plot,
            )

            #! format: off
            @info @sprintf "[lesson = %-20.20s] [epoch = %04i] Average training loss = %.2e\n" name epoch mean(training_losses)
            @info @sprintf "[lesson = %-20.20s] [epoch = %04i] Validation loss = %.2e\n" name epoch val_loss
            @info @sprintf "[lesson = %-20.20s] [epoch = %04i] Valid time = %.1f seconds\n" name epoch val_valid_time
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
                    epoch_duration,
                ],
            )

            early_stopping(val_loss) && @goto complete_training  # Use goto and label to break out of nested loops

            if val_valid_time > min_val_valid_time
                θ_min = copy(θ)
                min_val_epoch = epoch
                min_val_loss = val_loss
                min_val_valid_time = val_valid_time
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
    test_loss, test_valid_time =
        evaluate(prob, θ, test_data, loss, solver, reltol, abstol; maxiters, show_plot)

    @info "Training complete."
    @info @sprintf "Minimum validation loss = %.2e\n" min_val_loss
    @info @sprintf "Validation valid time = %.1f seconds\n" min_val_valid_time
    @info @sprintf "Test loss = %.2e\n" test_loss
    @info @sprintf "Test valid time = %.1f seconds\n" test_valid_time
    @info @sprintf "Training duration = %.1f seconds\n" training_duration

    return training_duration,
    learning_curve,
    epoch,
    min_val_epoch,
    min_val_loss,
    min_val_valid_time,
    test_loss,
    test_valid_time
end

function train!(
    θ::Vector{T},
    prob::SciMLBase.AbstractDEProblem,
    data::Data{T},
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
    norm = L2,
    regularisation_param::T = 0.0f0,
    # Early Stopping
    patience = Inf,
    time_limit = 23 * 60 * 60.0f0,
    # I/O
    verbose = false,
    show_plot = false,
) where {T<:AbstractFloat}
    # 1. Set up the loss function
    loss = (pred, target, θ) -> MSE(pred, target) + regularisation_param * norm(θ)

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
        patience,
        time_limit,
        verbose,
        show_plot,
    )
end
