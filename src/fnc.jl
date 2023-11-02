export ohmic_bcf
export exp_bcf

using SpecialFunctions

Γ = gamma

# generell definition

"""
the bath correlation function for an (sub-) Ohmic spectral density, i.e,
α(t) = 1/π ∫ dω η ω^s exp(-ω / wc) exp(-i ω t) = η/π Γ(s+1) ( wc/(1.0 + im*tau) )^(s+1)
"""
ohmic_bcf(t, s, η=1.0, wc=1.0) = η/π*Γ(s+1) * ( wc/(1.0 + im*t) )^(s+1)


"""
a reduced form of the (sub-) Ohmic BCF with only 's' as parameter and α(0) = 1
"""
red_ohmic_bcf(t, s) = (1. + im*t)^(-s-1)


"""
derivative of red_ohmic_bcf (autodiff does not work for complex functions :/ )
"""
ddt_red_ohmic_bcf(t, s) = -(s+1)*im*(1. + im*t)^(-s-2)


"""
a bath correlation function representation as a sum of exponentials, i.e.
α(t) = ∑_i G_i exp(-W_i t)

Params:
    G, W (complex) vectors holding the parameters
"""
function exp_bcf(t, G, W)
    sum(@. G*exp(-W*t))
end


"""derivative of exp_bcf"""
function ddt_exp_bcf(t, G, W)
    sum(@. -W*G*exp(-W*t) )
end


"""convert complex calued vectors (G,W) to real valued vector u"""
function GW2u(G::Vector{Complex{T}}, W::Vector{Complex{T}}) where {T <: Real}
    n = length(G) 
    u = Vector{T}(undef, 4n)
    a = -log.(G)
    u[1:4:end] .= real(a)
    u[2:4:end] .= imag(a)
    u[3:4:end] .= real(W)
    u[4:4:end] .= imag(W)
    return u
end

"""convert the real valued vector u to complex calued vectors (G,W)"""
function u2GW(u)
    a = complex.(u[1:4:end], u[2:4:end])
    G = exp.(-a)
    W = complex.(u[3:4:end], u[4:4:end])
    
    return G, W
end

struct ExpBCF_Loop end
struct ExpBCF_Dot end

"""
For optimzation we need the exp_bcf with real valued parameters, here u.

The first 4 values correpond to the first exponential term, i.e.
a1 = u[1] + im u[2]
b1 = u[3] + im u[4]
α(t) = exp(-(a1 + b1 * t)) + …
and so on.

Benchmarking autodiff has shown that a simple loop implementation
of the sum in combination with ForwardDiff is most efficient.
So we define the default exp_bcf with real parameters u to be
the loop implementation `exp_bcf_loop`.
"""
function exp_bcf(t, u)
    exp_bcf(t, u, ExpBCF_Loop())
end

"""
exponential bcf with real parameters u (@. definition)
"""
function exp_bcf(t, u, ::ExpBCF_Dot)
    G, W = u2GW(u)
    return exp_bcf(t, G, W)
end


"""
exponential bcf with real parameters u 
(straight forward loop implementation)
"""
function exp_bcf_loop(t, u)
    n = Integer(length(u) / 4)

    a = Complex(u[1], u[2])
    b = Complex(u[3], u[4])
    r = exp(-(a + b*t))
    for i in 1:n-1
        a = Complex(u[4i+1], u[4i+2])
        b = Complex(u[4i+3], u[4i+4])
        r += exp(-(a + b*t))
    end
    return r
end

exp_bcf(t, u, ::ExpBCF_Loop) = exp_bcf_loop(t, u)


"""detivative of the exponential representaion"""
function ddt_exp_bcf(t, u)
    n = Integer(length(u) / 4)
    a = Complex(u[1], u[2])
    b = Complex(u[3], u[4])
    r = -b*exp(-(a+b*t))
    for i in 1:n-1
        a = Complex(u[4i+1], u[4i+2])
        b = Complex(u[4i+3], u[4i+4])
        r += -b*exp(-(a+b*t))
    end
    return r
end

