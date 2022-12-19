function train!(
    prob,
    θ,
    data::Data{T};
    # Solver args
    solver::OrdinaryDiffEqAlgorithm = Tsit5(),
    reltol::T = 1.0f-6,
    abstol::T = 1.0f-6,
    maxiters = 10_000,
    sensealg::Union{Symbol,Nothing} = :BacksolveAdjoint,
    vjp::Union{Symbol,Nothing} = :ZygoteVJP,
    checkpointing::Bool = false,
    # Optimiser args
    optimiser_type::Type{O} = Adam,
    initial_learning_rate::T = 1.0f-3,
    min_learning_rate::T = 1.0f-3,
    decay_rate::Union{T,Nothing} = nothing,
    # Training args
    training_steps::AbstractVector = [1],
    epochs_per_step = 512,
    norm = L2,
    regularisation_param::T = 0.0f0,
    patience = Inf,
    time_limit = 23 * 60 * 60.0f0,
    prolong_training = false,
    initial_gc_interval = 0,
    # I/O args
    callback = display_progress,
    verbose = false,
    show_plot = false,
) where {T<:AbstractFloat,O<:Flux.Optimise.AbstractOptimiser}
    @info "Beginning training..."

    (; train_data, val_data, test_data) = data

    loss = (pred, target, θ) -> MSE(pred, target) + regularisation_param * norm(θ)

    # Set up the scheduled optimiser
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

    # Set up the adjoint sensitivity algorithm for computing gradients of the ODE solve
    sensealg = get_sensealg(sensealg, vjp, checkpointing)

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
    for steps_to_predict in training_steps
        data_loader = DataLoader(train_data, steps_to_predict)
        gc_interval = max(initial_gc_interval ÷ steps_to_predict, 1)

        if prolong_training && (steps_to_predict == training_steps[end])
            epochs_per_step = typemax(epochs_per_step)  # Train forever (or until the time limit) on the last training step
        end

        step_start_time = time()
        for _ = 1:epochs_per_step
            epoch += 1
            training_losses = Float32[]

            @info @sprintf "[epoch = %04i] [steps = %02i] Learning rate = %.1e" epoch steps_to_predict optimiser.flux_optimiser.eta

            iter = 0
            epoch_start_time = time()
            for (times, target_trajectory) in data_loader
                iter += 1
                tspan = (times[1], times[end])
                u0 = target_trajectory[:, 1]
                prob = remake(prob; u0, tspan)

                local predicted_trajectory, training_loss  # Declare local so we can access them outside of the following do block

                gradients = Zygote.gradient(Zygote.Params([θ])) do
                    retcode, predicted_trajectory = predict(
                        prob,
                        θ;
                        solver,
                        saveat = times,
                        reltol,
                        abstol,
                        maxiters,
                        sensealg,
                    )
                    training_loss = loss(predicted_trajectory, target_trajectory, θ)
                    return training_loss
                end

                !isnothing(callback) && callback(
                    epoch,
                    iter,
                    steps_to_predict,
                    tspan,
                    training_loss,
                    target_trajectory,
                    predicted_trajectory,
                    times;
                    verbose,
                    show_plot,
                )

                push!(training_losses, training_loss)
                Flux.update!(optimiser.flux_optimiser, θ, gradients[θ])

                # Call the garbace collector manually to avoid OOM errors on the cluster
                (initial_gc_interval != 0) && (iter % gc_interval == 0) && GC.gc(false)
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
            @info @sprintf "[epoch = %04i] [steps = %02i] Average training loss = %.2e\n" epoch steps_to_predict mean(training_losses)
            @info @sprintf "[epoch = %04i] [steps = %02i] Validation loss = %.2e\n" epoch steps_to_predict val_loss
            @info @sprintf "[epoch = %04i] [steps = %02i] Valid time = %.1f seconds\n" epoch steps_to_predict val_valid_time
            @info @sprintf "[epoch = %04i] [steps = %02i] Epoch duration = %.1f seconds\n" epoch steps_to_predict epoch_duration
            #! format: on

            push!(
                learning_curve,
                [
                    epoch,
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
                min_val_epoch = epoch
                min_val_loss = val_loss
                min_val_valid_time = val_valid_time
            end

            isa(optimiser, AbstractScheduledOptimiser) && update_learning_rate!(optimiser)

            if (time() - training_start_time) > time_limit
                #! format: off
                @info @sprintf "[epoch = %04i] [steps = %02i] Time limit of %.1f hours reached for the training loop. Stopping here." epoch steps_to_predict (time_limit / 3600)
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

    return learning_curve,
    min_val_epoch,
    min_val_loss,
    min_val_valid_time,
    test_loss,
    test_valid_time,
    training_duration
end

function predict(prob, θ; solver, saveat, reltol, abstol, maxiters, sensealg = nothing)
    sol = solve(prob, solver; p = θ, saveat, reltol, abstol, maxiters, sensealg)
    return sol.retcode, Array(sol)
end
