function eval_string(s)
    return eval(Meta.parse(s))
end

function ArgParse.parse_item(::Type{Symbol}, symbol_string::AbstractString)
    return Symbol(symbol_string)
end

function ArgParse.parse_item(::Type{Function}, function_name::AbstractString)
    return eval_string(function_name)
end

function ArgParse.parse_item(::Type{DataType}, type_name::AbstractString)
    return eval_string(type_name)
end

function ArgParse.parse_item(::Type{OrdinaryDiffEqAlgorithm}, solver_name::AbstractString)
    return eval_string(solver_name * "()")
end

function ArgParse.parse_item(::Type{Vector{T}}, arg_string::AbstractString) where {T}
    return [parse(T, item) for item in split(arg_string, ',')]
end

function get_common_settings()
    common_settings = ArgParseSettings(autofix_names = true)

    @add_arg_table common_settings begin
        # Experiment args
        "--job-id"
            arg_type = String
            default = get(ENV, "SLURM_JOB_ID", "1")
        "--rng-seed", "--seed"
            arg_type = Int
            default = 1

        # Neural net args
        "--hidden-layers", "--layers"
            arg_type = Int
            required = true
        "--hidden-width", "--width"
            arg_type = Int
            required = true
        "--activation"
            arg_type = Function
            default = relu

        # Training args
        "--norm"
            arg_type = Function
            default = L2
        "--regularisation-param", "--reg-param"
            arg_type = Float32
            default = 0f0
        "--optimiser-type", "--opt"
            arg_type = DataType
            default = Adam
        "--learning-rate", "--lr"
            arg_type = Float32
            default = 1f-2
        "--min-learning-rate", "--min-lr"
            arg_type = Float32
            default = 1f-2
        "--decay-rate", "--decay"
            arg_type = Float32
            default = 1f0
        "--epochs-per-step", "--epochs"
            arg_type = Int
            default = 1024
        "--patience"
            arg_type = Int
            default = typemax(Int)  # ~Inf
        "--training-steps", "--steps"
            arg_type = Vector{Int}
            default = [1]
        "--time-limit", "--time"
            arg_type = Float32
            default = Inf32
        "--prolong-training", "--prolong"
            action = :store_true
        "--initial-gc-interval", "--gc-interval"
            arg_type = Int
            default = 0

        # Solver args
        "--reltol"
            arg_type = Float32
            default = 1f-6
        "--abstol"
            arg_type = Float32
            default = 1f-6
        "--solver"
            arg_type = OrdinaryDiffEqAlgorithm
            default = Tsit5()
        "--maxiters"
            arg_type = Int
            default = 10_000
        "--sensealg"
            arg_type = Union{Symbol,Nothing}
            default = nothing
        "--autojacvec"
            arg_type = Union{Symbol,Nothing}
            default = nothing

        # I/0
        "--verbose"
            action = :store_true
        "--show-plot"
            action = :store_true
        "--results-file"
            arg_type = String
            default = "results.csv"
        "--model-dir"
            arg_type = String
            default = "models"
        "--learning-curve-dir", "--lc-dir"
            arg_type = String
            default = "learning_curves"
    end

    return common_settings
end

function parse_command_line()
    common_settings = get_common_settings()
    return parse_args(common_settings)
end
