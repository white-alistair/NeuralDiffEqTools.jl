abstract type AbstractOptimiser{T} end
abstract type ScheduledOptimiser{T} <: AbstractOptimiser{T} end

struct ExpDecayOptimiser{O<:Flux.Optimise.AbstractOptimiser,T<:AbstractFloat} <:
       ScheduledOptimiser{T}
    flux_opt::O
    eta_min::T
    decay_rate::T
end

function update_learning_rate!(optimiser)
    (; flux_opt, eta_min, decay_rate) = optimiser
    flux_opt.eta = max(eta_min, flux_opt.eta * decay_rate)
    return nothing
end
