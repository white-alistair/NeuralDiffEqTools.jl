struct Lesson{T,O<:AbstractOptimiser{T}}
    name::String
    steps::Int
    epochs::Int
    optimiser::O
end

struct Curriculum{T,O<:Optimisers.Leaf}
    name::String
    hash::UInt64
    optimiser_state::O  # Shared across all lessons
    lessons::Vector{Lesson{T}}
end

function Curriculum(curriculum_file, opt_state, T)
    curriculum_dict = TOML.parsefile(curriculum_file)
    name = curriculum_dict["name"]

    lessons = Lesson{T}[]
    for lesson_dict in curriculum_dict["lessons"]
        name, epochs, steps, schedule = unpack_lesson(lesson_dict)
        optimiser = get_optimiser(opt_state, epochs, schedule, T)
        lesson = Lesson{T,typeof(optimiser)}(name, steps, epochs, optimiser)
        push!(lessons, lesson)
    end

    return Curriculum{T,typeof(opt_state)}(name, hash(curriculum_dict), opt_state, lessons)
end

function unpack_lesson(lesson_dict)
    name = lesson_dict["name"]
    epochs = lesson_dict["epochs"]
    steps = lesson_dict["steps"]
    schedule = lesson_dict["schedule"]
    return name, epochs, steps, schedule
end

function get_optimiser(opt_state, epochs, schedule, T)
    schedule_type = schedule["type"]
    if schedule_type == "constant"
        learning_rate = schedule["learning_rate"]
        return ConstantLearningRateOptimiser{typeof(opt_state),T}(opt_state, learning_rate)
    else
        initial_learning_rate = schedule["initial_learning_rate"]
        final_learning_rate = schedule["final_learning_rate"]
        if schedule_type == "linear_warmup"
            return LinearWarmupOptimiser(
                opt_state,
                initial_learning_rate,
                final_learning_rate,
                epochs,
            )
        elseif schedule_type == "linear_decay"
            return LinearDecayOptimiser(
                opt_state,
                initial_learning_rate,
                final_learning_rate,
                epochs,
            )
        elseif schedule_type == "exponential_decay"
            return ExponentialDecayOptimiser(
                opt_state,
                initial_learning_rate,
                final_learning_rate,
                epochs,
            )
        end
    end
end
