abstract type AbstractOptimiser{T} end
abstract type AbstractScheduledOptimiser{T} <: AbstractOptimiser{T} end

struct ExponentialDecayOptimiser{O<:Flux.Optimise.AbstractOptimiser,T<:AbstractFloat} <:
       AbstractScheduledOptimiser{T}
    flux_optimiser::O
    min_learning_rate::T
    decay_rate::T
end

function ExponentialDecayOptimiser(
    optimiser_type::Type{O},
    initial_learning_rate::T,
    min_learning_rate::T,
    decay_rate::T,
) where {T<:AbstractFloat,O<:Flux.Optimise.AbstractOptimiser} 
    flux_optimiser = optimiser_type(initial_learning_rate)
    return ExponentialDecayOptimiser{O,T}(flux_optimiser, min_learning_rate, decay_rate)
end

function ExponentialDecayOptimiser(
    optimiser_type::Type{O},
    initial_learning_rate::T,
    min_learning_rate::T,
    epochs::Integer,
) where {T<:AbstractFloat,O<:Flux.Optimise.AbstractOptimiser}
    flux_optimiser = optimiser_type(initial_learning_rate) 
    decay_rate = (min_learning_rate / initial_learning_rate)^(1 / epochs)
    return ExponentialDecayOptimiser{O,T}(flux_optimiser, min_learning_rate, decay_rate)
end

function update_learning_rate!(optimiser::ExponentialDecayOptimiser)
    (; flux_optimiser, min_learning_rate, decay_rate) = optimiser
    flux_optimiser.eta = max(min_learning_rate, flux_optimiser.eta * decay_rate)
    return nothing
end
