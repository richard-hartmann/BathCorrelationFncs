"""
    Here we want to figure out which AutoDiff module is best suited for the
    p-norm minimuzation problem.
"""

using Revise
using BCF
using BenchmarkTools
using Optimization
using ForwardDiff
using ReverseDiff
# using Tracker           # was superseded by zygote
using Zygote
using FiniteDiff
using CairoMakie
# using ModelingToolkit   # appers to be a little more involved to use


function sq_sum(u)
    return sum(u .^ 2)
end

struct UseForwardDiff end
struct UseReverseDiff end
#struct UseTracker end
struct UseZygote end
struct UseFiniteDiff end
# struct UseModelingToolkit end

all_ad_types = [UseForwardDiff(), UseReverseDiff(), UseZygote(), UseFiniteDiff()]

function gradient(f, x, ::UseForwardDiff)
    ForwardDiff.gradient(f, x)
end

function gradient(f, x, ::UseReverseDiff)
    ReverseDiff.gradient(f, x)
end

function gradient(f, x, ::UseZygote)
    first(Zygote.gradient(f, x))
end

function gradient(f, x, ::UseFiniteDiff)
    FiniteDiff.finite_difference_gradient(f, x)
end

function simple_test()
    x = Float64[1,2]
    @assert gradient(sq_sum, x, UseFiniteDiff()) ≈ [2, 4] atol = 1e-8
    @assert gradient(sq_sum, x, UseForwardDiff()) ≈ [2, 4] atol = 1e-8
    @assert gradient(sq_sum, x, UseReverseDiff()) ≈ [2, 4] atol = 1e-8
    @assert gradient(sq_sum, x, UseZygote()) ≈ [2, 4] atol = 1e-8
    println("all simple tests passed")
    
end
simple_test()


"""
    Time how long it takes to calculate the
    gradient for the p-norm diff between a refecence bath correlation
    function and the exponential representation.
"""
function benchmark_grads(f_diff, ad, range_of_exp_terms)
    tau = 0:0.1:5
    f1_tau = BCF.red_ohmic_bcf.(tau, 1)
    p = 5

    times = Vector{Float64}(undef, length(range_of_exp_terms))

    for (i, num_exp_terms) in enumerate(range_of_exp_terms)
        param = tau, f1_tau, p, num_exp_terms
        f_opt = u -> f_diff(u, param)

        u0 = rand(4num_exp_terms)
        gradient(f_opt, u0, ad)

        n_runs = 10
        tot_time = 0.
        for i in range(1, n_runs)
            u0 = rand(4num_exp_terms)
            s = @timed gradient(f_opt, u0, ad)
            tot_time += s.time
        end
        tot_time /= n_runs
        times[i] = tot_time
    end

    return times
end

n_range = 5:30

fig = Figure(resolution=(600, 1000))

for (i, ad) in enumerate(all_ad_types)
    println("run $ad")
    ax = Axis(fig[i, 1], yscale=log10, title = "$ad")
    limits!(ax, first(n_range), last(n_range), 1e-5, 1e-1)

    tms = benchmark_grads(
        (t, u) -> BCF.diff_exp(t, u, BCF.ExpBCF_Dot()), 
        ad, 
        n_range
    )
    lines!(ax, n_range, tms, label="dot")
    scatter!(ax, n_range, tms, label="dot")

    tms = benchmark_grads(
        (t, u) -> BCF.diff_exp(t, u, BCF.ExpBCF_Loop()), 
        ad, 
        n_range
    )
    lines!(ax, n_range, tms, label="loop")
    scatter!(ax, n_range, tms, label="loop")

    tms = benchmark_grads(
        BCF._diff_ohmic_exp_full_loop, 
        ad, 
        n_range
    )
    lines!(ax, n_range, tms, label="full loop")
    scatter!(ax, n_range, tms, label="full loop")

    axislegend(ax, merge=true, position=:rb)
end


fig

println("UseForwardDiff and exp_bcf in a loop fashion seems best")

save("./dev/benchmark_autodiff.pdf", fig)



