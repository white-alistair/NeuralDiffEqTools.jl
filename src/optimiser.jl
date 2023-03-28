function get_optimiser(rule_type, hyperparameters)
    if rule_type == :Adam
        rule = Optimisers.Adam()
    elseif rule_type == :AdamW
        rule = Optimisers.AdamW()
    end
    return Optimisers.adjust(rule; hyperparameters...)
end
