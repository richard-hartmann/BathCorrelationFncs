using Revise

using BCF

s = 0.3

ts = LinRange(0, 10, 5)
u_exact = BCF.get_n_time_exp_repr_for_ahmic_bcf(ts, s)

for ti in ts
    a_exp = BCF.exp_bcf(ti, u_exact)
    a_ref = BCF.red_ohmic_bcf(ti, s)
    println("diff alpha: $(abs(a_exp - a_ref))")

    ddt_a_exp = BCF.ddt_exp_bcf(ti, u_exact)
    ddt_a_ref = BCF.ddt_red_ohmic_bcf(ti, s)
    println("diff ddt_ alpha: $(abs(ddt_a_exp - ddt_a_ref))")

end
