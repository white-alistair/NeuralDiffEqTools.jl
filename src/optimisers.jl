abstract type AbstractOptimiser{T} end
abstract type AbstractScheduledOptimiser{T} <: AbstractOptimiser{T} end

function setup_optimiser(rule::Optimisers.AbstractRule, θ::AbstractArray{T}) where {T}
    return Optimisers.setup(rule, θ)
end

# Constant learning rate
struct ConstantLearningRateOptimiser{R<:Optimisers.AbstractRule,T<:AbstractFloat} <:
       AbstractOptimiser{T}
    rule::R
    learning_rate::T
end

function ConstantLearningRateOptimiser{R,T}(rule, learning_rate) where {R,T}
    rule = Optimisers.adjust(rule; eta = learning_rate)
    return ConstantLearningRateOptimiser{R,T}(rule, learning_rate)
end

# Exponentially decaying learning rate
struct ExponentialDecayOptimiser{R<:Optimisers.AbstractRule,T<:AbstractFloat} <:
       AbstractScheduledOptimiser{T}
    rule::R
    initial_learning_rate::T
    decay_rate::T
end

function ExponentialDecayOptimiser{R,T}(
    rule::Optimisers.AbstractRule,
    initial_learning_rate,
    final_learning_rate,
    epochs,
) where {R,T}
    rule = Optimisers.adjust(rule; eta = initial_learning_rate)
    decay_rate = (final_learning_rate / initial_learning_rate)^(1 / (epochs - 1))
    return ExponentialDecayOptimiser{R,T}(rule, initial_learning_rate, decay_rate)
end

function update_learning_rate!(
    opt_state::Optimisers.Leaf,
    optimiser::ExponentialDecayOptimiser,
)
    old_learning_rate = get_learning_rate(opt_state.rule)
    new_learning_rate = old_learning_rate * optimiser.decay_rate
    Optimisers.adjust!(opt_state; eta = new_learning_rate)
    return nothing
end

# Linearly decaying learning rate
struct LinearDecayOptimiser{R<:Optimisers.AbstractRule,T<:AbstractFloat} <:
       AbstractScheduledOptimiser{T}
    rule::R
    initial_learning_rate::T
    step::T
end

function LinearDecayOptimiser{R,T}(
    rule::Optimisers.AbstractRule,
    initial_learning_rate,
    final_learning_rate,
    epochs,
) where {R,T}
    rule = Optimisers.adjust(rule; eta = initial_learning_rate)
    step = (initial_learning_rate - final_learning_rate) / (epochs - 1)
    return LinearDecayOptimiser{R,T}(rule, initial_learning_rate, step)
end

function update_learning_rate!(
    opt_state::Optimisers.Leaf,
    optimiser::LinearDecayOptimiser{T},
) where {T}
    old_learning_rate = get_learning_rate(opt_state.rule)
    new_learning_rate = old_learning_rate - optimiser.step
    Optimisers.adjust!(opt_state; eta = new_learning_rate)
    return nothing
end

# Linear Warmup
struct LinearWarmupOptimiser{R<:Optimisers.AbstractRule,T<:AbstractFloat} <:
       AbstractScheduledOptimiser{T}
    rule::R
    initial_learning_rate::T
    step::T
end

function LinearWarmupOptimiser{R,T}(
    rule::Optimisers.AbstractRule,
    initial_learning_rate,
    final_learning_rate,
    epochs,
) where {R,T}
    rule = Optimisers.adjust(rule; eta = initial_learning_rate)
    step = (final_learning_rate - initial_learning_rate) / (epochs - 1)
    return LinearWarmupOptimiser{R,T}(rule, initial_learning_rate, step)
end

function update_learning_rate!(opt_state::Optimisers.Leaf, optimiser::LinearWarmupOptimiser)
    old_learning_rate = get_learning_rate(opt_state.rule)
    new_learning_rate = old_learning_rate + optimiser.step
    Optimisers.adjust!(opt_state; eta = new_learning_rate)
    return nothing
end

# Other methods
function get_learning_rate(rule::Optimisers.AbstractRule)
    return rule.eta
end

function get_learning_rate(rule::Optimisers.OptimiserChain)
    return rule.opts[1].eta  # For AdamW
end
