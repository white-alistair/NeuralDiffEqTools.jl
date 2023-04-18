function predict(
    prob,
    θ;
    solver = Tsit5(),
    saveat,
    reltol = 1.0f-6,
    abstol = 1.0f-6,
    maxiters = 100_000,
    sensealg = nothing,
)
    sol = solve(prob, solver; p = θ, saveat, reltol, abstol, maxiters, sensealg)
    return sol.t, Array(sol)
end
