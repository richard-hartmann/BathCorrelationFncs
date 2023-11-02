using Revise
using BCF

t_max = 100
s = 0.8
num_exp_terms = 5
p = 5
u_init_min = [ 0,   0,  0,   0]
u_init_max = [10, 2pi, 10, 2pi]
num_samples = 3

#sols, i = BCF.sobol_scan_fit(t_max, s, num_exp_terms, p, u_init_min, u_init_max, num_samples; num_tau=150, a_tilde=0.1, u_tilde=0.5, skip=0, maxiters=10000)


id = (t_max, s, num_exp_terms, p, u_init_min, u_init_max, num_samples, 150, 0.1, 0.5, 0, 10000)
hash(id)



a = [1,2,3,4]
using Serialization


struct MyS4{T}
    a::T
    b::T
end

struct MyS2
    a
    b
end

a = Int8(3)
b = Int8(2)

mys = MyS4{Int64}(a, b)



mys2 = MyS2(a, a)

hash(mys)
hash(mys2)

serialize("mydata", mys)

deserialize("mydata")


function te(::Type{Val(:rel_diff)}, a)
    println("use rel diff $a")
end

function te(a)
    println("use fallback $a")
end

function te(_t, a)
    println(typeof(_t))
    te(a)
end

te(_t::Symbol, args...) = te(Val{_t}, args...)
    
symbs = [:other, :rel_diff]

kind = rand(symbs)
te(kind, 5)

te(2)


Type{Val(:rel_diff)}

methods(te)


methods(diff)


a = :a
a



s = :mysymb

q = s

q === :mysymb

s = 7