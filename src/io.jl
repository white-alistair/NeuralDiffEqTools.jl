function write_results(filename; kwargs...)
    # If the file doesn't exist already, create it and add the header row.
    if !isfile(filename)
        col_names = [keys(kwargs)]
        open(filename, "w") do io
            writedlm(io, col_names, ',')
        end
    end

    # Format float values.
    formatted_cols = []
    for col in values(kwargs)
        if col isa AbstractFloat
            push!(formatted_cols, @sprintf "%.2e" col)
        elseif col isa AbstractVector
            push!(formatted_cols, string(col))
        else
            push!(formatted_cols, col)
        end
    end

    # Write the data to the file.
    sleep(60 * rand())  # Hack to avoid concurrent writes
    open(filename, "a") do io
        writedlm(io, [formatted_cols], ',')
    end

    return nothing
end
