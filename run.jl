using InteractiveUtils
versioninfo(stderr)

using NeuralDiffEqTools, Parameters

args = Manifolds.parse_command_line()

# Log command line args
ordered_args = sort(collect(args), by = x -> x[1])
for (arg_name, arg_value) in ordered_args
    @info "$arg_name = $arg_value"
end

# Unpack command line args into variables in current scope
@unpack job_id, rng_seed = args
# Solver args
@unpack reltol, abstol, solver, maxiters = args
# Neural net args
@unpack hidden_layers, hidden_width, activation = args
# Training args
@unpack norm, regularisation_param, optimiser, learning_rate, min_learning_rate,
decay_rate, epochs_per_step, patience, training_steps, time_limit, prolong_training,
initial_gc_interval = args
# I/0
@unpack verbose, show_plot, results_file, model_dir, learning_curve_dir = args

using Random
Random.seed!(rng_seed)

# Generate training data
data = Manifolds.generate_train_val_test_data(
    systembf,
    train_seconds,
    val_seconds,
    test_seconds,
    n_validation_sets,
    n_test_sets;
    transient_seconds,
    solver = data_solver,
    reltol = data_reltol,
    abstol = data_abstol,
    dt = BigFloat(dt),
)
u0 = data.train_data[1]

# Need to set up the model

# Train the model
@time learning_curve,
min_val_epoch,
val_loss,
val_valid_time,
test_loss,
test_valid_time,
duration = Manifolds.train!(
    model,
    data;
    solver,
    reltol,
    abstol,
    maxiters,
    optimiser,
    learning_rate,
    min_learning_rate,
    decay_rate,
    norm,
    regularisation_param,
    epochs_per_step,
    patience,
    time_limit,
    prolong_training,
    initial_gc_interval,
    training_steps,
    verbose,
    show_plot,
)

# I/O
Manifolds.write_results(
    results_file;
    job_id,
    system,
    problem,
    hidden_layers,
    hidden_width,
    norm,
    regularisation_param,
    manifold_method,
    manifold_param,
    val_loss,
    test_loss,
    val_valid_time,
    test_valid_time,
    min_val_epoch,
    activation,
    duration,
    time_limit,
    prolong_training,
    dt,
    transient_seconds,
    train_seconds,
    val_seconds,
    test_seconds,
    n_validation_sets,
    n_test_sets,
    training_steps,
    epochs_per_step,
    patience,
    optimiser,
    learning_rate,
    min_learning_rate,
    decay_rate,
    reltol,
    abstol,
    maxiters,
)

Manifolds.save_model(model, dir = model_dir)
Manifolds.save_learning_curve(learning_curve, model.id, dir = learning_curve_dir)
