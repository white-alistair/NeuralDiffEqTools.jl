struct Lesson{O<:Union{Optimisers.AbstractRule,Nothing},S<:ParameterSchedulers.Stateful}
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

function get_optimiser(lesson_dict)
    rule_type = get(lesson_dict, "optimiser", nothing)
    if isnothing(rule_type)
        return nothing
    elseif rule_type == "Adam"
        return Optimisers.Adam()
    elseif rule_type == "AdamW"
        rule = Optimisers.AdamW()
        weight_decay = lesson_dict["weight_decay"]
        return Optimisers.adjust(rule; gamma = weight_decay)
    end
end

function get_scheduler(lesson_dict)
    schedule_type = lesson_dict["schedule"]
    if schedule_type == "Constant"
        @unpack lr = lesson_dict
        return ParameterSchedulers.Stateful(ParameterSchedulers.Constant(lr))
    elseif schedule_type == "ExpDecay"
        @unpack min_lr, max_lr, epochs = lesson_dict
        decay_rate = (min_lr / max_lr)^(1 / (epochs - 1))
        return ParameterSchedulers.Stateful(
            ParameterSchedulers.Exp(; λ = max_lr, γ = decay_rate),
        )
    elseif schedule_type == "CosAnneal"
        @unpack min_lr, max_lr = lesson_dict
        period = get(lesson_dict, "period", lesson_dict["epochs"])  # Use period if given, else epochs
        return ParameterSchedulers.Stateful(
            ParameterSchedulers.CosAnneal(; λ0 = max_lr, λ1 = min_lr, period),
        )
    elseif schedule_type == "LinearRamp"
        @unpack min_lr, max_lr, epochs = lesson_dict
        return ParameterSchedulers.Stateful(
            ParameterSchedulers.Triangle(; λ0 = max_lr, λ1 = min_lr, period = 2 * epochs),
        )
    end
end
