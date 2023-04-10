function eval_string(s)
    return eval(Meta.parse(s))
end

function ArgParse.parse_item(::Type{Function}, function_name::AbstractString)
    return eval_string(function_name)
end

function ArgParse.parse_item(::Type{OrdinaryDiffEqAlgorithm}, solver_name::AbstractString)
    return eval_string(solver_name * "()")
end

function ArgParse.parse_item(::Type{NamedTuple}, arg_string::AbstractString)
    items = eachsplit(arg_string, ",")
    names = [Symbol(strip.(split(item, "="))[1]) for item in items]
    args = [parse(Float32, (split(item, "=")[2])) for item in items]
    return (; zip(names, args)...)
end

function get_common_settings()
    common_settings = ArgParseSettings(; autofix_names = true)

    #! format: off
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
        "--epochs"
            arg_type = Int
            required = true
        "--schedule-file", "--schedule"
            arg_type = String
            required = true
        "--optimiser-rule", "--opt"
            arg_type = Symbol
            default = :Adam
        "--optimiser-hyperparams", "--opt-params"
            arg_type = NamedTuple
            default = (;)
        "--patience"
            arg_type = Int
            default = typemax(Int)  # ~Inf
        "--time-limit", "--time"
            arg_type = Float32
            default = Inf32
        "--n-manual-gc"
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
        "--sensealg"
            arg_type = Symbol
            default = :BacksolveAdjoint
        "--vjp"
            arg_type = Symbol
            default = :ReverseDiffVJP
        "--checkpointing"
            action = :store_true

        # I/0
        "--verbose"
            action = :store_true
        "--show-plot"
            action = :store_true
        "--results-file"
            arg_type = String
            default = "results.csv"
        "--learning-curve-dir", "--lc-dir"
            arg_type = String
            default = "learning_curves"
    end
    #! format: on

    return common_settings
end

function parse_command_line()
    common_settings = get_common_settings()
    return parse_args(common_settings)
end

function log_args(args)
    ordered_args = sort(collect(args); by = x -> x[1])
    for (arg_name, arg_value) in ordered_args
        @info "$arg_name = $arg_value"
    end
end
