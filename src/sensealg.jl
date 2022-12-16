function get_sensealg(sensealg, vjp, checkpointing)
    if isnothing(sensealg)
        return nothing
    end

    if isnothing(vjp)
        autojacvec = nothing
    elseif vjp == :ZygoteVJP
        autojacvec = ZygoteVJP()
    elseif vjp == :ReverseDiffVJP
        autojacvec = ReverseDiffVJP(true)
    end
    
    if sensealg == :BacksolveAdjoint
        return BacksolveAdjoint(; autojacvec, checkpointing)
    elseif sensealg == :InterpolatingAdjoint
        return InterpolatingAdjoint(; autojacvec, checkpointing)
    elseif sensealg == :QuadratureAdjoint
        return QuadratureAdjoint(; autojacvec)  # Doesn't work with ZygoteVJP for out-of-place problems
    end
end
