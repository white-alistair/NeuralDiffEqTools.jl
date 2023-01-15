import Optimisers: update!

abstract type AbstractOptimiser{T} end
abstract type AbstractScheduledOptimiser{T} <: AbstractOptimiser{T} end

struct ConstantLearningRateOptimiser{O<:Optimisers.Leaf,T<:AbstractFloat} <:
       AbstractOptimiser{T}
    state::O
    learning_rate::T
end

struct ExponentialDecayOptimiser{O<:Optimisers.Leaf,T<:AbstractFloat} <:
       AbstractScheduledOptimiser{T}
    state::O
    initial_learning_rate::T
    min_learning_rate::T
    decay_rate::T
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

function update_learning_rate!(opt::ExponentialDecayOptimiser)
    (; state, min_learning_rate, decay_rate) = opt
    old_learning_rate = state.rule.eta
    new_learning_rate = max(min_learning_rate, old_learning_rate * decay_rate)
    Optimisers.adjust!(state; eta = new_learning_rate)
    return nothing
end

struct LinearDecayOptimiser{O<:Optimisers.Leaf,T<:AbstractFloat} <:
       AbstractScheduledOptimiser{T}
    state::O
    initial_learning_rate::T
    min_learning_rate::T
    decay::T
end

function LinearDecayOptimiser(
    state::Optimisers.Leaf,
    initial_learning_rate::T,
    min_learning_rate::T,
    decay::Union{T,Nothing},
    epochs::Union{Int,Nothing},
) where {T}
    if isnothing(decay)
        decay = (initial_learning_rate - min_learning_rate)/(epochs - 1)
    end

    return LinearDecayOptimiser{typeof(state),Float32}(
        state,
        initial_learning_rate,
        min_learning_rate,
        decay,
    )
end

function update_learning_rate!(opt::LinearDecayOptimiser)
    (; state, min_learning_rate, decay) = opt
    old_learning_rate = state.rule.eta
    new_learning_rate = max(min_learning_rate, old_learning_rate - decay)
    Optimisers.adjust!(state; eta = new_learning_rate)
    return nothing
end

function Optimisers.update!(opt::AbstractOptimiser, θ, Δθ)
    Optimisers.update!(opt.state, θ, Δθ[1])
end

function set_initial_learning_rate!(opt::ConstantLearningRateOptimiser)
    (; state, learning_rate) = opt
    Optimisers.adjust!(state; eta = learning_rate)
    return nothing
end

function set_initial_learning_rate!(opt::AbstractScheduledOptimiser)
    (; state, initial_learning_rate) = opt
    Optimisers.adjust!(state; eta = initial_learning_rate)
    return nothing
end

function get_learning_rate(opt::AbstractOptimiser)
    return opt.state.rule.eta
end
