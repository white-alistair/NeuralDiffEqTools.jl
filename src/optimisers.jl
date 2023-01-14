import Optimisers: update!

abstract type AbstractOptimiser{T} end
abstract type AbstractScheduledOptimiser{T} <: AbstractOptimiser{T} end

struct ExponentialDecayOptimiser{O<:Optimisers.Leaf,T<:AbstractFloat} <:
       AbstractScheduledOptimiser{T}
    state::O
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
        min_learning_rate,
        decay_rate,
    )
end

function Optimisers.update!(opt::AbstractOptimiser, θ, Δθ)
    Optimisers.update!(opt.state, θ, Δθ[1])
end

function update_learning_rate!(optimiser::ExponentialDecayOptimiser)
    (; state, min_learning_rate, decay_rate) = optimiser
    old_learning_rate = state.rule.eta
    new_learning_rate = max(min_learning_rate, old_learning_rate * decay_rate)
    Optimisers.adjust!(state; eta = new_learning_rate)
    return nothing
end

function get_learning_rate(opt::AbstractScheduledOptimiser)
    return opt.state.rule.eta
end
