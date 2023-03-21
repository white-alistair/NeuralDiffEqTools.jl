struct Lesson{T,O<:AbstractOptimiser{T}}
    name::String
    steps::Int
    epochs::Int
    optimiser::O
end

struct Curriculum{T}
    name::String
    hash::UInt64
    lessons::Vector{Lesson{T}}
end

function Curriculum(curriculum_file, T)
    curriculum_dict = TOML.parsefile(curriculum_file)
    curriculum_name = curriculum_dict["name"]

    lessons = Lesson{T}[]
    for lesson_dict in curriculum_dict["lessons"]
        lesson_name, epochs, steps, optimiser_dict = unpack_lesson(lesson_dict)
        optimiser = get_optimiser(optimiser_dict, epochs, T)
        lesson = Lesson{T,typeof(optimiser)}(lesson_name, steps, epochs, optimiser)
        push!(lessons, lesson)
    end

    return Curriculum{T}(
        curriculum_name,
        hash(curriculum_dict),
        lessons,
    )
end

function unpack_lesson(lesson_dict)
    name = lesson_dict["name"]
    epochs = lesson_dict["epochs"]
    steps = lesson_dict["steps"]
    optimiser_dict = lesson_dict["optimiser"]
    return name, epochs, steps, optimiser_dict
end

function get_opt_rule(opt_dict)
    rule_type = opt_dict["rule"]
    if rule_type == "Adam"
        return Optimisers.Adam()
    elseif rule_type == "AdamW"
        rule = Optimisers.AdamW()
        weight_decay = opt_dict["weight_decay"]
        return Optimisers.adjust(rule, gamma = weight_decay)
    end
end

function get_optimiser(opt_dict, epochs, T)
    opt_rule = get_opt_rule(opt_dict)
    schedule_type = opt_dict["schedule_type"]
    if schedule_type == "constant"
        learning_rate = opt_dict["learning_rate"]
        return ConstantLearningRateOptimiser{typeof(opt_rule),T}(opt_rule, learning_rate)
    else
        initial_learning_rate = opt_dict["initial_learning_rate"]
        final_learning_rate = opt_dict["final_learning_rate"]
        if schedule_type == "linear_warmup"
            return LinearWarmupOptimiser{typeof(opt_rule),T}(
                opt_rule,
                initial_learning_rate,
                final_learning_rate,
                epochs,
            )
        elseif schedule_type == "linear_decay"
            return LinearDecayOptimiser{typeof(opt_rule),T}(
                opt_rule,
                initial_learning_rate,
                final_learning_rate,
                epochs,
            )
        elseif schedule_type == "exponential_decay"
            return ExponentialDecayOptimiser{typeof(opt_rule),T}(
                opt_rule,
                initial_learning_rate,
                final_learning_rate,
                epochs,
            )
        end
    end
end
