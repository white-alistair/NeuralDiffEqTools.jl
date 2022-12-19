function predict(prob, θ; solver, saveat, reltol, abstol, maxiters, sensealg = nothing)
    sol = solve(prob, solver; p = θ, saveat, reltol, abstol, maxiters, sensealg)
    return sol.retcode, Array(sol)
end
