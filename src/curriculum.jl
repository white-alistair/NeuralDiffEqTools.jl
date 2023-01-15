struct Lesson{T,O<:AbstractOptimiser{T}}
    name::String
    steps::Int
    epochs::Int
    optimiser::O
end

struct Curriculum{T,O<:Optimisers.Leaf}
    opt_state::O
    lessons::Vector{Lesson{T}}
end

function init_curriculum(curriculum_file, θ::AbstractVector{T}) where {T}
    curriculum_dict = TOML.parsefile(curriculum_file)
    opt_state = get_optimiser_state(curriculum_dict, θ)

    lessons = Lesson{T}[]
    for lesson_dict in curriculum_dict["lessons"]
        name, epochs, steps, schedule_dict = unpack_lesson(lesson_dict)
        initial_learning_rate, min_learning_rate, decay_rate =
            unpack_schedule(schedule_dict)

        optimiser = ExponentialDecayOptimiser(
            opt_state,
            initial_learning_rate,
            min_learning_rate,
            decay_rate,
            epochs,
        )

        lesson = Lesson{T,typeof(optimiser)}(name, steps, epochs, optimiser)
        push!(lessons, lesson)
    end

    return Curriculum(opt_state, lessons)
end

function get_optimiser_state(curriculum_dict, θ)
    rule_type = curriculum_dict["optimiser"]["rule"]
    if rule_type == "Adam"
        rule = Optimisers.Adam()  # Default params (these will be changed later)
    end
    state = Optimisers.setup(rule, θ)
    return state
end

function unpack_lesson(lesson_dict)
    name = lesson_dict["name"]
    epochs = lesson_dict["epochs"]
    steps = lesson_dict["steps"]
    schedule_dict = lesson_dict["schedule"]
    return name, epochs, steps, schedule_dict
end

function unpack_schedule(schedule_dict)
    initial_learning_rate = schedule_dict["initial_learning_rate"]
    min_learning_rate = schedule_dict["min_learning_rate"]
    decay_rate = get(schedule_dict, "decay_rate", nothing)
    return initial_learning_rate, min_learning_rate, decay_rate
end
