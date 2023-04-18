function evaluate(prob, θ, data, loss, solver, reltol, abstol)
    losses = Float32[]
    for (times, target_trajectory) in data
        tspan = (times[1], times[end])
        u0 = target_trajectory[:, 1]
        prob = remake(prob; u0, tspan)
        _, predicted_trajectory = predict(
            prob,
            θ;
            saveat = times,
            solver,
            reltol,
            abstol,
        )
        push!(losses, loss(predicted_trajectory, target_trajectory))
    end
    return mean(losses)
end
