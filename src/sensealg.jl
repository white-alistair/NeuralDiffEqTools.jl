function get_sensealg(sensealg, vjp)
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
        return BacksolveAdjoint(; autojacvec)
    elseif sensealg == :InterpolatingAdjoint
        return InterpolatingAdjoint(; autojacvec)
    elseif sensealg == :QuadratureAdjoint
        return QuadratureAdjoint(; autojacvec)  # Doesn't work with ZygoteVJP
    end
end
