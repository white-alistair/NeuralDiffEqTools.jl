function get_optimiser(rule_type, hyperparameters; clip_norm = nothing)
    if rule_type == :Adam
        rule = Optimisers.Adam()
    elseif rule_type == :AdamW
        rule = Optimisers.AdamW()
    elseif rule_type == :Nesterov
        rule = Optimisers.Nesterov()
    end

    if !isempty(hyperparameters)
        rule = Optimisers.adjust(rule; hyperparameters...)
    end

    if !isnothing(clip_norm)
        rule = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_norm), rule)
    end

    return rule
end
