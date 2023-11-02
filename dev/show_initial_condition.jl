using BCF
using CairoMakie

include("conv_plotting.jl")


tau = 0:0.1:5
s = 1
al_tau = BCF.red_ohmic_bcf.(tau, s)

f = Figure(resolution = (1200, 1200))
ax = Axis(f[1,1:2], title="local exp fit")
ax_accum = Axis(f[2,1], title="accum of naive exp fits")
ax_accum_err = Axis(f[2,2], yscale=log10, title="... and abs error")
ax_accum2 = Axis(f[3,1], title="accum of exp fits")
ax_accum2_err = Axis(f[3,2], yscale=log10, title="... and abs error")

cplx_lines!(ax, tau, al_tau, label="ref", color=:black)
cplx_lines!(ax_accum, tau, al_tau, label="ref", color=:black)

exp_aprx = zeros(eltype(al_tau), size(al_tau)...)
exp2_aprx = zeros(eltype(al_tau), size(al_tau)...)



for tau_0 in [4.5, 2.5, 0.5]
    local u = BCF.fit_single_exp_to_ohmic_at_t(tau_0, s)
    local al_exp_tau = [BCF.exp_bcf(ti, u) for ti in tau]
    cplx_lines!(ax, tau, al_exp_tau, label="match at $tau_0")

    global exp_aprx += al_exp_tau
    lines!(ax_accum_err, tau, abs.(al_tau - exp_aprx))
    cplx_lines!(ax_accum, tau, exp_aprx)

end

ylims!(ax_accum_err, 1e-8, 1e2)

f[:, 3] = Legend(f, ax, framevisible = false)
save("./inital_cond.pdf", f)

f

u = (1,2,3,4)

x = Matrix{Float64}(undef, 0, 4)
v = Matrix{Float64}(undef, 1, 4)
print(v[1,:])
v[1,:] = u
#vcat(x, u)



