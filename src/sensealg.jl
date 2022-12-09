function get_sensealg(sensealg_type, vjp_type)
    if isnothing(sensealg_type)
        return nothing
    end

    if isnothing(vjp_type)
        autojacvec = nothing
    if vjp_type == :ZygoteVJP
        autojacvec = ZygoteVJP()
    elseif vjp_type == :ReverseDiffVJP
        autojacvec = ReverseDiffVJP(true)
    end
    
    if sensealg_type == :BacksolveAdjoint
        return BacksolveAdjoint(; autojacvec)
    elseif sensealg_type == :InterpolatingAdjoint
        return InterpolatingAdjoint(; autojacvec)
    elseif sensealg_type == :QuadratureAdjoint
        return QuadratureAdjoint(; autojacvec)
    end
end
