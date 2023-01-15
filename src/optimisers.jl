import Optimisers: update!

abstract type AbstractOptimiser{T} end
abstract type AbstractScheduledOptimiser{T} <: AbstractOptimiser{T} end

struct ExponentialDecayOptimiser{O<:Optimisers.Leaf,T<:AbstractFloat} <:
       AbstractScheduledOptimiser{T}
    state::O
    initial_learning_rate::T
    min_learning_rate::T
    decay_rate::T
end

function ExponentialDecayOptimiser(
    rule_type::Type{O},
    θ::AbstractVector{T},
    initial_learning_rate::T,
    min_learning_rate::T,
    decay_rate::Union{T,Nothing},
    epochs::Union{Int,Nothing},
) where {T<:AbstractFloat,O<:Optimisers.AbstractRule}
    rule = rule_type(initial_learning_rate)
    state = Optimisers.setup(rule, θ)

    if isnothing(decay_rate)
        decay_rate = (min_learning_rate / initial_learning_rate)^(1 / (epochs - 1))
    end

    return ExponentialDecayOptimiser{typeof(state),T}(
        state,
        initial_learning_rate,
        min_learning_rate,
        decay_rate,
    )
end

function ExponentialDecayOptimiser(
    state::Optimisers.Leaf,
    initial_learning_rate::T,
    min_learning_rate::T,
    decay_rate::Union{T,Nothing},
    epochs::Union{Int,Nothing},
) where {T}
    if isnothing(decay_rate)
        decay_rate = (min_learning_rate / initial_learning_rate)^(1 / (epochs - 1))
    end

    return ExponentialDecayOptimiser{typeof(state),Float32}(
        state,
        initial_learning_rate,
        min_learning_rate,
        decay_rate,
    )
end

function Optimisers.update!(opt::AbstractOptimiser, θ, Δθ)
    Optimisers.update!(opt.state, θ, Δθ[1])
end

function set_initial_learning_rate!(opt::AbstractScheduledOptimiser)
    (; state, initial_learning_rate) = opt
    Optimisers.adjust!(state; eta = initial_learning_rate)
    return nothing
end

function update_learning_rate!(opt::ExponentialDecayOptimiser)
    (; state, min_learning_rate, decay_rate) = opt
    old_learning_rate = state.rule.eta
    new_learning_rate = max(min_learning_rate, old_learning_rate * decay_rate)
    Optimisers.adjust!(state; eta = new_learning_rate)
    return nothing
end

function get_learning_rate(opt::AbstractScheduledOptimiser)
    return opt.state.rule.eta
end
