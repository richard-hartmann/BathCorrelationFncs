using BCF
using CairoMakie
using Printf

include("conv_plotting.jl")

fig = Figure(resolution=(800,600))


ax = Axis(fig[1,1], title="BCF")
ax_err = Axis(
    fig[2,1], 
    title="error of the exp repr.",
    yscale=log10,
    #limits = (nothing, nothing, 1e-10, 1)
)
s = 1
t_max = 100

t = 0.01:0.01:t_max

bcf_t = BCF.red_ohmic_bcf.(t, s)
cplx_lines!(ax, t, bcf_t, color=:black, label="ref")
lines!(ax_err, t, abs.(bcf_t), color = :black, lw=3)

ts = [0.1, 1, 2, 4, 8]
n_ts = length(ts)

r = nothing

i = 1

while true
    global i
    if i % 1000 == 0
        print("\r$(i)k runs ...")
    end
    u_guess = Vector{Float64}(undef, 4*n_ts)
    u_guess[1:4:end] = 100*rand(Float64, n_ts)  # e^-u1
    u_guess[2:4:end] = 2pi*rand(Float64, n_ts)  # e^-(i u2)
    u_guess[3:4:end] = 10*rand(Float64, n_ts)  # e^-(u3 t)
    u_guess[4:4:end] = 2pi*rand(Float64, n_ts)  # e^-(i u4 t)

    global r
    r = BCF.solve_multi_exp_for_ohmic_bcf(ts, u_guess, s)

    if BCF.converged(r)
        println("\nrun $i has converged")
        break
    end

    i += 1
end

u_exact = r.zero
println("decay rates $(u_exact[3:4:end])")

exp_bcf_t = BCF.exp_bcf.(t, Ref(u_exact))
lr, lc = cplx_lines!(ax, t, exp_bcf_t)
vlines!(ax, ts, color=:gray, ls=:dash)
lines!(ax_err, t, abs.(bcf_t - exp_bcf_t))

fig[1,2] = Legend(fig, ax)
fig

