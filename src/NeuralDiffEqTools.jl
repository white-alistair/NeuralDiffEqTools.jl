module NeuralDiffEqTools

using OrdinaryDiffEq,
    SciMLBase,
    SciMLSensitivity,
    Flux,
    Zygote,
    Optimisers,
    ArgParse,
    LinearAlgebra,
    Statistics,
    Random,
    Printf,
    DelimitedFiles,
    TOML,
    JLD2,
    CairoMakie

include("time_series.jl")
include("data.jl")
include("multiple_shooting.jl")
include("prob.jl")
include("neural_nets.jl")
include("optimisers.jl")
include("curriculum.jl")
include("adjoints.jl")
include("losses.jl")
include("predict.jl")
include("valid_time.jl")
include("evaluate.jl")
include("train.jl")
include("io.jl")
include("learning_curves.jl")
include("command_line.jl")


end
