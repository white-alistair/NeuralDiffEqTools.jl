function MSE(
    predicted_trajectory,
    target_trajectory,
)
    return mean(abs2, predicted_trajectory[:, 2:end] .- target_trajectory[:, 2:end])  # Do not include u0
end
