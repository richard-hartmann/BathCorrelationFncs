using Revise

using BCF
using Test


###########################################################
# simple tests of the exp bcf

G = [2.0 + 0im]
W = [0.0 + 1im]

@test BCF.exp_bcf(0, G, W) == 2
@test BCF.exp_bcf(1, G, W) == 2 * exp(-1im)
@test BCF.exp_bcf(pi, G, W) ≈ -2 atol=1e-12

G = [1.0 + im, 2.3 - 4im]
W = [0.3 - im, 0.4 - 2im]

u = BCF.GW2u(G, W)
for t in 0:0.1:5
    @test BCF.exp_bcf(2, u) ≈ BCF.exp_bcf(2, G, W) atol=1e-10
end

@test BCF.exp_bcf(2, u) ≈ BCF.exp_bcf(2, u, BCF.ExpBCF_Loop()) atol=1e-10
@test BCF.exp_bcf(2, u) ≈ BCF.exp_bcf(2, u, BCF.ExpBCF_Dot()) atol=1e-10
@test BCF.exp_bcf(2, u) ≈ BCF.exp_bcf_loop(2, u) atol=1e-10


###########################################################
# match exp bcf to ohmic bcf

s = 0.8

tau_0 = 1
# verify that smooth_single_exp_to_ohmic_bcf_at_t returns parameters
# for a single exponential that matches smoothly the Ohmic bcf (parameter s)
# at a given time point tau_0
u = BCF.smooth_single_exp_to_ohmic_bcf_at_t(tau_0, s)
@test BCF.red_ohmic_bcf(tau_0, s) ≈ BCF.exp_bcf(tau_0, u) atol = 1e-10
@test BCF.ddt_red_ohmic_bcf(tau_0, s) ≈ -complex(u[3], u[4])*BCF.exp_bcf(tau_0, u) atol=1e-10

# verify that the residual of that exponential at tau_0 vanishes
F = Vector{Float64}(undef, 4)
BCF._residual_solve_multi_exp_for_ohmic_bcf!(F, u, [tau_0], s)
@test F ≈ zeros(length(F)) atol=1e-10

# here we check that the NL-solver yields the same exp function, i.e., same u
u0 = Float64[1,1,1,1]
sol_NL = BCF.solve_multi_exp_for_ohmic_bcf([tau_0], u0, s)
@test sol_NL.zero ≈ u atol=1e-8


###########################################################
# test TauExpRange

n = 16001
t_max = 1500
al_tilde = 0.1
for u_tilde in [0.4, 0.5, 0.8]
    tau0, a = BCF.calc_tau_0_a(t_max, s, al_tilde, u_tilde)
    # final value (u=1)
    local u = 1
    @test tau0*(exp(a*u) - 1) ≈ t_max atol=1e-8
end

for s in [0.3, 0.8, 1.6]
    for u_tilde in [0.2, 0.5, 0.8]
        tr = get_TauExpRange(t_max, s, n, al_tilde, u_tilde)
        @test length(tr) == n
        tr = collect(tr)
        @test length(tr) == n
        @test tr[1] == 0                  # initial value (u=0)
        @test tr[end] ≈ t_max atol=1e-8   # final value (u=1)
        tau_u = tr[Integer(floor(u_tilde*n))]
        @test abs(BCF.red_ohmic_bcf(tau_u, s)) ≈ al_tilde atol=al_tilde/10
    end
end

###########################################################
# minimize difference between exp bcf and ohmic bcf

@test iszero(BCF.diff(:abs_p_diff, [1,2], ([1,2], 2)))
@test iszero(BCF.diff(:abs_p_diff, [1,2], ([1,2], 5.67)))


d1 = 2
d2 = 3.4
# abs diff
@test BCF.diff(:abs_p_diff, [2,5im], ([2+d1,(5+d2)*1im], 3)) ≈ (d1^3 + d2^3)^(1/3) atol=1e-12
# relative diff, actually weighted diff, here with weights 1.0 -> so same result as abs diff
@test BCF.diff(:rel_p_diff, [2,5im], ([2+d1,(5+d2)*1im], [1.0, 1.0], 3)) ≈ (d1^3 + d2^3)^(1/3) atol=1e-12

@test_throws MethodError BCF.diff(:new_diff, [2,5im], ([2+d1,(5+d2)*1im], [1.0, 1.0], 3))


n = 150
t_max = 100
s = 0.5
tr = get_TauExpRange(t_max, s, n)
tr = collect(tr)
bcf_tau = [BCF.red_ohmic_bcf(ti, s) for ti in tr]
p = 5
u0 = [1,1,1,1, 0.5, 0.5, 0.5, 0.5]
method = BCF.NelderMead()
maxiters = 1000

sol = BCF.fit_ohmic_exp(tr, bcf_tau, p, u0, method, maxiters, :abs_p_diff)
@test Integer(sol.retcode) == 1
@test sol.objective ≈ 0.05585703941513527 atol=1e-10

sol = BCF.fit_ohmic_exp(tr, bcf_tau, p, u0, method, maxiters, :rel_p_diff)
@test Integer(sol.retcode) == 1
@test sol.objective ≈ 0.979426791250466 atol=1e-10

using BCF

t_max = 100
s = 0.5
num_exp_terms = 4
p = 5
u_init_min = (0, 0, 0, 0)
u_init_max = (10, 2pi, 10, 2pi)
diff_kind = :abs_p_diff

ofc = BCF.OhmicFitCfg(t_max, s, num_exp_terms, p, u_init_min, u_init_max, diff_kind)
r = repr(ofc)
@test hash(r) == 5685999255080489228
ofs = BCF.OhmicFitState(ofc)

ofc2 = BCF.OhmicFitCfg(t_max, s, num_exp_terms, p, u_init_min, u_init_max, diff_kind)
@test ofc == ofc2

ofc16 = BCF.OhmicFitCfg{Float16}(t_max, s, num_exp_terms, p, u_init_min, u_init_max, diff_kind)
r = repr(ofc16)
@test hash(r) == 11410535683856806746
ofs16 = BCF.OhmicFitState(ofc16)

@test ofc != ofc16



# cunrch the first 5 samples
ofs = BCF.OhmicFitState(ofc)
cnt_new = BCF.sobol_scan_fit!(ofc, ofs, 5)
@test ofs.solutions[1].idx == 1
@test ofs.solutions[end].idx == 4
@test ofs.sobol_idx == 5
@test cnt_new == length(ofs.solutions)

# continue until we have 10 samples
cnt_new2 = BCF.sobol_scan_fit!(ofc, ofs, 10)
@test cnt_new + cnt_new2 == length(ofs.solutions)
@test ofs.sobol_idx == 10

path = "test_fits"
fname = BCF.ofc_identifyer(ofc)
fullname = joinpath(path, fname)
ispath(fullname) && rm(fullname)
c = BCF.fit(ofc, 10, path, verbose=false)
@test c == 10

c = BCF.fit(ofc, 10, path, verbose=false)
@test c == 0

println("all tests passed")

