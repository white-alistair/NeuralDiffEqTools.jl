struct Lesson{T,O<:AbstractOptimiser{T}}
    name::String
    steps::Int
    epochs::Int
    optimiser::O
end

struct Curriculum{T,O<:Optimisers.Leaf}
    optimiser_state::O  # Shared across all lessons
    lessons::Vector{Lesson{T}}
end

function Curriculum(curriculum_file, opt_state, T)
    curriculum_dict = TOML.parsefile(curriculum_file)

    lessons = Lesson{T}[]
    for lesson_dict in curriculum_dict["lessons"]
        name, epochs, steps, schedule = unpack_lesson(lesson_dict)
        optimiser = get_optimiser(opt_state, epochs, schedule, T)
        lesson = Lesson{T,typeof(optimiser)}(name, steps, epochs, optimiser)
        push!(lessons, lesson)
    end

    return Curriculum{T,typeof(opt_state)}(opt_state, lessons)
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
    if schedule_type == "linear_warmup"
        initial_learning_rate = schedule["initial_learning_rate"]
        final_learning_rate = schedule["final_learning_rate"]
        return LinearWarmupOptimiser{typeof(opt_state),T}(
            opt_state,
            initial_learning_rate,
            final_learning_rate,
            epochs,
        )
    elseif schedule_type == "constant"
        learning_rate = schedule["learning_rate"]
        return ConstantLearningRateOptimiser{typeof(opt_state),T}(opt_state, learning_rate)
    elseif schedule_type == "linear_decay"
        initial_learning_rate = schedule["initial_learning_rate"]
        min_learning_rate = schedule["min_learning_rate"]
        decay = get(schedule, "decay", nothing)
        return LinearDecayOptimiser(
            opt_state,
            initial_learning_rate,
            min_learning_rate,
            decay,
            epochs,
        )
    elseif schedule_type == "exponential_decay"
        initial_learning_rate = schedule["initial_learning_rate"]
        min_learning_rate = schedule["min_learning_rate"]
        decay_rate = get(schedule, "decay_rate", nothing)
        return ExponentialDecayOptimiser(
            opt_state,
            initial_learning_rate,
            min_learning_rate,
            decay_rate,
            epochs,
        )
    end
end
