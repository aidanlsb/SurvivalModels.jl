using SurvivalModels, Distributions, DataFramesMeta, PythonCall, ForwardDiff, DiffRules
using LinearAlgebra, ProgressMeter
using LogExpFunctions

function sim_mixture_cure(dist, c; N=10_000, thresh=1.0)
    cured = rand(Bernoulli(c), N)
    x = Array{Float64}(undef, N)
    observed = Array{Bool}(undef, N)
    for (i, c) in enumerate(cured)
        if c
            observed[i] = false       
            x[i] = thresh
        else
            d = rand(dist)
            if d > thresh
                observed[i] = false
                x[i] = thresh
            else
                observed[i] = true
                x[i] = d
            end
        end
    end
    return DataFrame(x=x, observed=observed)
end

df = sim_mixture_cure(Weibull(0.40, 130945), 0.96; N=100_000, thresh=30_000)
mcb = MixtureCureEstimator(WeibullEstimator())
mc = SurvivalModels.fit(mcb, df.x, df.observed)


import SurvivalModels: neg_log_likelihood, cumulative_hazard, log_hazard

params = [logistic(-3.0), exp(-0.9), exp(5.02)]
nll = neg_log_likelihood(mcb, df.x, df.observed, params)

ci = confint(mc, df.x, df.observed)


ll = pyimport("lifelines")
mcpy = ll.MixtureCureFitter(base_fitter=ll.WeibullFitter())
mcpy.fit(df.x, event_observed=df.observed)

println("Cured fraction from lifelines is $(1 - mcpy.cured_fraction_) and from this package is $(logistic(mc.cured_fraction))")
println("α from lifelines is $(mcpy.rho_) and from this package is $(exp(mc.base_estimator.α))")
println("θ from lifelines is $(mcpy.lambda_) and from this package is $(exp(mc.base_estimator.θ))")


function run_sim()
    c_real = 0.9
    alpha_real = -0.7
    theta_real = 3.0
    df = sim_mixture_cure(Weibull(exp(alpha_real), exp(theta_real)), 1 - logistic(c_real); N=10_000, thresh=100)
    mcb = MixtureCureEstimator(WeibullEstimator())
    mc = SurvivalModels.fit(mcb, df.x, df.observed)

    lo, hi = confint(mc, df.x, df.observed)

    c_covered = lo[1] .<= c_real .<= hi[1]
    alpha_covered = lo[2] .<= alpha_real .<= hi[2]
    theta_covered = lo[3] .<= theta_real .<= hi[3]
    covered = [c_covered, alpha_covered, theta_covered]
    return covered
end

N_sims = 1_000
estimates = Array{Float64}(undef, N_sims, 3)
p = Progress(N_sims)
Threads.@threads for i in 1:N_sims
    covered_sim = run_sim()
    estimates[i, :] = covered_sim
    next!(p)
end
finish!(p)

p_c = mean(estimates[:, 1])
println("Coverage for c is $p_c")
p_alpha = mean(estimates[:, 2])
println("Coverage for alpha is $p_alpha")
p_theta = mean(estimates[:, 3])
println("Coverage for theta is $p_theta")

