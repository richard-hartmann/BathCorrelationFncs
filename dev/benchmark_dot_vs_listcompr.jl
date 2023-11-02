using BenchmarkTools

function my_func(x, vector_of_parameters)
    x*vector_of_parameters[1] + vector_of_parameters[2]
end

function my_func_2(x, a, b)
    x*a + b
end

function call_dot_syntax_Ref(x_array, vector_of_parameters)
    # reduce my_func to a single argument function
    y = my_func.(x_array, [vector_of_parameters])
    return y
end


function call_dot_syntax(x_array, vector_of_parameters)
    # reduce my_func to a single argument function
    _f = x -> my_func(x, vector_of_parameters)
    y = _f.(x_array)
    return y
end

function call_list_compr(x_array, vector_of_parameters)
    y = [my_func(xi, vector_of_parameters) for xi in x_array]
    return y
end

function call_dot_syntax_2(x_array, vector_of_parameters)
    a = vector_of_parameters[1]
    b = vector_of_parameters[2]
    y = my_func_2.(x_array, a, b)
    return y
end

function call_loop(x_array, vector_of_parameters)
    y = similar(x_array)
    for (i, x) in enumerate(x_array)
        y[i] = my_func(x, vector_of_parameters)
    end
    return y
end


x_array = rand(5000)
vector_of_parameters = Float64[2, 3]

println("#"^24)
println("dot syntax with anonymous function to shadow vector character of p")
t = @benchmark call_dot_syntax(x_array, vector_of_parameters)
println(median(t))

println("#"^24)
println("use list comprehension")
t = @benchmark call_list_compr(x_array, vector_of_parameters)
println(median(t))

println("#"^24)
println("straight forward dot syntax with Ref(p) to shadow vector character of p")
t = @benchmark call_dot_syntax_Ref(x_array, vector_of_parameters)
println(median(t))

println("#"^24)
println("use straight forward dot syntax with expanded parameters, i.e., p -> (a, b)")
t = @benchmark call_dot_syntax_2(x_array, vector_of_parameters)
println(median(t))

println("#"^24)
println("use for loop")
t = @benchmark call_loop(x_array, vector_of_parameters)
println(median(t))