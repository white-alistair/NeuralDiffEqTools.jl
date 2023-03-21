struct Curriculum
    name::String
    hash::UInt64
    lessons::Vector{Lesson}
end

function Curriculum(curriculum_dict::Dict{String,Any})
    curriculum_name = curriculum_dict["name"]
    lessons = [Lesson(lesson_dict) for lesson_dict in curriculum_dict["lessons"]]
    return Curriculum(curriculum_name, hash(curriculum_dict), lessons)
end

struct Lesson{O<:Optimisers.AbstractRule,S<:ParameterSchedulers.Stateful}
    name::String
    steps::Int
    epochs::Int
    optimiser::O
    scheduler::S
end

function Lesson(lesson_dict::Dict{String,Any})
    @unpack name, steps, epochs = lesson_dict
    optimiser = get_optimiser(lesson_dict)
    scheduler = get_scheduler(lesson_dict)
    return Lesson(name, steps, epochs, optimiser, scheduler)
end

function get_optimiser(lesson_dict)
    rule_type = lesson_dict["optimiser"]
    if rule_type == "Adam"
        return Optimisers.Adam()
    elseif rule_type == "AdamW"
        rule = Optimisers.AdamW()
        weight_decay = lesson_dict["weight_decay"]
        return Optimisers.adjust(rule; gamma = weight_decay)
    end
end

function get_scheduler(lesson_dict)
    schedule_type = lesson_dict["schedule"]
    if schedule_type == "CosAnneal"
        @unpack max_lr, min_lr, epochs = lesson_dict
        return ParameterSchedulers.Stateful(
            CosAnneal(; λ0 = max_lr, λ1 = min_lr, period = epochs),
        )
    end
end
