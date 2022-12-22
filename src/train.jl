function train!(
    θ::Vector{T},
    data::Data{T},
    loss::Function,
    optimiser::AbstractOptimiser{T},
    prob::SciMLBase.AbstractDEProblem,
    solver::SciMLBase.AbstractDEAlgorithm = Tsit5(),
    adjoint::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm = BacksolveAdjoint(;
        autojacvec = ReverseDiffVJP(true),
    );
    reltol::T = 1.0f-6,
    abstol::T = 1.0f-6,
    maxiters = 10_000,
    batch_size = 1,
    training_steps::AbstractVector = [1],
    epochs_per_step = 512,
    patience = Inf,
    time_limit = 23 * 60 * 60.0f0,
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

    epochs = 0
    training_start_time = time()
    for steps_to_predict in training_steps
        step_start_time = time()
        for _ = 1:epochs_per_step
            epochs += 1

            @info @sprintf "[epoch = %04i] [steps = %02i] Learning rate = %.1e" epochs steps_to_predict optimiser.flux_optimiser.eta

            iter = 0
            training_losses = Float32[]
            epoch_start_time = time()
            for (batch_times, batch_trajectories) in DataLoader(train_data, steps_to_predict, batch_size)
                iter += 1

                local training_loss

                gradients = Zygote.gradient(Zygote.Params([θ])) do
                    N = 0
                    training_loss = 0

                    for (target_times, target_trajectory) in zip(batch_times, batch_trajectories)
                        tspan = (target_times[1], target_times[end])
                        u0 = target_trajectory[:, 1]
                        prob = remake(prob; u0, tspan)

                        retcode, predicted_trajectory = predict(
                            prob,
                            θ;
                            solver,
                            saveat = target_times,
                            reltol,
                            abstol,
                            maxiters,
                            sensealg = adjoint,
                        )

                        N += 1
                        training_loss += loss(predicted_trajectory, target_trajectory, θ)
                    end

                    return training_loss / N
                end

                push!(training_losses, training_loss)
                Flux.update!(optimiser.flux_optimiser, θ, gradients[θ])
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
            @info @sprintf "[epoch = %04i] [steps = %02i] Average training loss = %.2e\n" epochs steps_to_predict mean(training_losses)
            @info @sprintf "[epoch = %04i] [steps = %02i] Validation loss = %.2e\n" epochs steps_to_predict val_loss
            @info @sprintf "[epoch = %04i] [steps = %02i] Valid time = %.1f seconds\n" epochs steps_to_predict val_valid_time
            @info @sprintf "[epoch = %04i] [steps = %02i] Epoch duration = %.1f seconds\n" epochs steps_to_predict epoch_duration
            #! format: on

            push!(
                learning_curve,
                [
                    epochs,
                    steps_to_predict,
                    optimiser.flux_optimiser.eta,
                    mean(training_losses),
                    val_loss,
                    epoch_duration,
                ],
            )

            early_stopping(val_loss) && @goto complete_training  # Use goto and label to break out of nested loops

            if val_loss < min_val_loss
                θ_min = copy(θ)
                min_val_epoch = epochs
                min_val_loss = val_loss
                min_val_valid_time = val_valid_time
            end

            isa(optimiser, AbstractScheduledOptimiser) && update_learning_rate!(optimiser)

            if (time() - training_start_time) > time_limit
                #! format: off
                @info @sprintf "[epoch = %04i] [steps = %02i] Time limit of %.1f hours reached for the training loop. Stopping here." epochs steps_to_predict (time_limit / 3600)
                @goto complete_training  # Use goto and label to break out of nested loops
                #! format: on
            end

            flush(stderr)  # Keep log files up to date on the cluster
        end
        step_duration = time() - step_start_time
        @info @sprintf "[steps = %02i] Step duration = %.1f seconds\n" steps_to_predict step_duration
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
    epochs,
    min_val_epoch,
    min_val_loss,
    min_val_valid_time,
    test_loss,
    test_valid_time
end

function train!(
    θ::Vector{T},
    data::Data{T},
    prob::SciMLBase.AbstractDEProblem;
    # Solver
    solver::SciMLBase.AbstractDEAlgorithm = Tsit5(),
    reltol::T = 1.0f-6,
    abstol::T = 1.0f-6,
    maxiters = 10_000,
    # Adjoint
    sensealg::Union{Symbol,Nothing} = :BacksolveAdjoint,
    vjp::Union{Symbol,Nothing} = :ReverseDiffVJP,
    checkpointing::Bool = false,
    # Optimiser
    optimiser_type::Type{O} = Adam,
    initial_learning_rate::T = 1.0f-3,
    min_learning_rate::T = 1.0f-3,
    decay_rate::Union{T,Nothing} = nothing,
    # Training Schedule
    batch_size::Int = 1,
    training_steps::AbstractVector = [1],
    epochs_per_step = 512,
    # Regularisation
    norm = L2,
    regularisation_param::T = 0.0f0,
    # Early Stopping
    patience = Inf,
    time_limit = 23 * 60 * 60.0f0,
    # I/O
    show_plot = false,
) where {T<:AbstractFloat,O<:Flux.Optimise.AbstractOptimiser}
    # 1. Set up the loss function
    loss = (pred, target, θ) -> MSE(pred, target) + regularisation_param * norm(θ)

    # 2. Set up the scheduled optimiser
    if isnothing(decay_rate)
        total_epochs = length(training_steps) * epochs_per_step
        optimiser = ExponentialDecayOptimiser(
            optimiser_type,
            initial_learning_rate,
            min_learning_rate,
            total_epochs,
        )
    else
        optimiser = ExponentialDecayOptimiser(
            optimiser_type,
            initial_learning_rate,
            min_learning_rate,
            decay_rate,
        )
    end

    # 3. Set up the adjoint sensitivity algorithm for computing gradients of the ODE solve
    adjoint = get_adjoint(sensealg, vjp, checkpointing)

    return train!(
        θ,
        data,
        loss,
        optimiser,
        prob,
        solver,
        adjoint;
        reltol,
        abstol,
        maxiters,
        batch_size,
        training_steps,
        epochs_per_step,
        patience,
        time_limit,
        show_plot,
    )
end
