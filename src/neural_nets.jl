function get_mlp((input_size, output_size)::Pair, hidden_layers, hidden_width, activation)
    mlp = Chain(
        Dense(input_size => hidden_width, activation),                                 # Input layer
        [Dense(hidden_width => hidden_width, activation) for _ = 1:hidden_layers]...,  # Hidden layers
        Dense(hidden_width => output_size),                                            # Output layer
    )
    θ, re = Flux.destructure(mlp)
    return θ, re
end
