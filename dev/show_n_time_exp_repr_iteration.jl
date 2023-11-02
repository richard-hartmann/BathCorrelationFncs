using BCF
using CairoMakie
using Printf

include("conv_plotting.jl")

fig = Figure(resolution=(800,600))


ax = Axis(fig[1,1], title="BCF")
ax_err = Axis(
    fig[2,1], 
    title="error of the exp repr.",
    #yscale=log10,
    #limits = (nothing, nothing, 1e-10, 1)
)
s = 1
t_max = 20
n_max = 8

t = 0.01:0.01:t_max

bcf_t = BCF.red_ohmic_bcf.(t, s)
cplx_lines!(ax, t, bcf_t, color=:black, label="ref")
lines!(ax_err, t, abs.(bcf_t), color = :black, lw=3)

#ts = LinRange(0, t_max, n_max)
#ts = [0, 1, 3, 10, 20]
ts = [0, 0.5, 1, 2, 4, 6, 10, 20]

#ts = [0.1, 0.2 ]


println("add t1 = $(ts[1])")
u_exact = BCF.smooth_single_exp_to_ohmic_bcf_at_t(ts[1], s)

exp_bcf_t = BCF.exp_bcf.(t, Ref(u_exact))

lr, lc = cplx_lines!(ax, t, exp_bcf_t)
vlines!(ax, ts[1], color=cl(lr), label="use t1 = $(ts[1])")
lines!(ax_err, t, abs.(bcf_t - exp_bcf_t))

for i in 2:length(ts)
    global u_exact
    print("add t$i = $(ts[i]) ")        
    # most trivial guess, completely independent of the parameters u that have already been determined
    # ui = BCF.smooth_single_exp_to_ohmic_bcf_at_t(ts[i], s)

    # here we get the exp tha is smooth in α(ti, s) - ∑ exp(ti ...)
    ui = BCF.smooth_single_exp_to_ohmic_bcf_minus_exp_at_t(ts[i], s, u_exact)
    
    u_guess = [u_exact ; ui]
    r = BCF.solve_multi_exp_for_ohmic_bcf(ts[1:i], u_guess, s)

    if ! BCF.converged(r)
        println("FAILED 🗲 (return latest good solution which has $(Integer(length(u_exact)/4)) exp terms")
        return u_exact
    end

    println("✓")
    u_exact = r.zero

    exp_bcf_t = BCF.exp_bcf.(t, Ref(u_exact))
    lr, lc = cplx_lines!(ax, t, exp_bcf_t)
    vlines!(ax, ts[i], color=cl(lr), label = (@sprintf "use t%i = %.1f" i ts[i]))
    lines!(ax_err, t, abs.(bcf_t - exp_bcf_t))
end

fig[1,2] = Legend(fig, ax)
fig

