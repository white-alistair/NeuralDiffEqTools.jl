abstract type AbstractOptimiser{T} end
abstract type AbstractScheduledOptimiser{T} <: AbstractOptimiser{T} end

struct ExponentialDecayOptimiser{O<:Flux.Optimise.AbstractOptimiser,T<:AbstractFloat} <:
       AbstractScheduledOptimiser{T}
    flux_optimiser::O
    min_learning_rate::T
    decay_rate::T
end

function update_learning_rate!(optimiser::ExponentialDecayOptimiser)
    (; flux_optimiser, min_learning_rate, decay_rate) = optimiser
    flux_optimiser.eta = max(min_learning_rate, flux_optimiser.eta * decay_rate)
    return nothing
end
