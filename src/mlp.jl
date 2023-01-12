function get_mlp(
    (input_size, output_size)::Pair,
    hidden_layers,
    hidden_width,
    activation = relu,
)
    mlp = Chain(
        Dense(input_size => hidden_width, activation),                                     # First hidden layer
        [Dense(hidden_width => hidden_width, activation) for _ = 1:(hidden_layers-1)]...,  # Remaining hidden layers
        Dense(hidden_width => output_size),                                                # Output layer
    )
    θ, re = Flux.destructure(mlp)
    return θ, re
end
