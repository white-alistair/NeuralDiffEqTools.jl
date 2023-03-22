function predict(prob, θ; solver, saveat, reltol, abstol, sensealg = nothing)
    sol = solve(prob, solver; p = θ, saveat, reltol, abstol, sensealg)
    return Array(sol)
end
