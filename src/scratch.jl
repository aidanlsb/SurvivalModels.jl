using SurvivalModels, Distributions, DataFramesMeta, PythonCall, ForwardDiff, DiffRules
using LinearAlgebra, ProgressMeter
using LogExpFunctions

function sim_mixture_cure(dist, c; N=10_000, thresh=1.0)
    cured = rand(Bernoulli(c), N)
    t = Array{Float64}(undef, N)
    observed = Array{Bool}(undef, N)
    for (i, c) in enumerate(cured)
        if c
            observed[i] = false       
            t[i] = thresh
        else
            d = rand(dist)
            if d > thresh
                observed[i] = false
                t[i] = thresh
            else
                observed[i] = true
                t[i] = d
            end
        end
    end
    return DataFrame(t=t, observed=observed)
end

function sim_mixture_cure_reg(β_c, β_α, β_θ; N=10_000, thresh=1.0)
    # cured = rand(Bernoulli(c), N)
    t = Array{Float64}(undef, N)
    x = rand(Normal(), N)
    observed = Array{Bool}(undef, N)
    α = exp.(0.5 .+ x .* β_α)
    θ = exp.(0.75 .+ x .* β_θ)
    c_ = logistic.(2.0 .+ x .* β_c)
    for i in 1:N
        c = rand(Bernoulli(c_[i]))
        if c 
            observed[i] = false
            t[i] = thresh
        else
            dist = Weibull(α[i], θ[i])
            d = rand(dist) 
            if d > thresh
                observed[i] = false
                t[i] = thresh
            else 
                observed[i] = true
                t[i] = d
            end
        end
    end
    df = DataFrame(t=t, x=x, observed=observed)
    df.intercept = ones(Float64, nrow(df))
    return df
end

df = sim_mixture_cure(Weibull(0.40, 130945), 0.96; N=100_000, thresh=30_000)
mcb = MixtureCureEstimator(WeibullEstimator())
mc = SurvivalModels.fit(mcb, df.t, df.observed)

# df_reg = sim_mixture_cure_reg([-0.25], [0.9], [0.45]; N=100_000, thresh=10)
df_reg = sim_mixture_cure_reg([2.3], [1.5], [-0.4]; N=100_000, thresh=10)
t = df_reg.t
e = df_reg.observed
println(mean(e))
X = Matrix(df_reg[:, [:x]])
mcr = SurvivalModels.fit(mcb, t, e, X)

import SurvivalModels: initialize_params, compute_params, transform_params

β0 = initialize_params(mcb, t, e, X)
βmat = compute_params(mcb, X, β0)
tb = transform_params(mcb, βmat)



mc_regular = SurvivalModels.fit(mcb, t, e)

# import SurvivalModels: neg_log_likelihood, cumulative_hazard, log_hazard
ci = confint(mc, df.t, df.observed)

ll = pyimport("lifelines")
mcpy = ll.MixtureCureFitter(base_fitter=ll.WeibullFitter())
mcpy.fit(df.t, event_observed=df.observed)

println("Cured fraction from lifelines is $(1 - mcpy.cured_fraction_) and from this package is $(logistic(mc.cured_fraction))")
println("α from lifelines is $(mcpy.rho_) and from this package is $(exp(mc.base_estimator.α))")
println("θ from lifelines is $(mcpy.lambda_) and from this package is $(exp(mc.base_estimator.θ))")


function run_sim()
    c_real = 0.9
    alpha_real = -0.7
    theta_real = 3.0
    df = sim_mixture_cure(Weibull(exp(alpha_real), exp(theta_real)), 1 - logistic(c_real); N=10_000, thresh=100)
    mcb = MixtureCureEstimator(WeibullEstimator())
    mc = SurvivalModels.fit(mcb, df.t, df.observed)

    lo, hi = confint(mc, df.t, df.observed)

    c_covered = lo[1] .<= c_real .<= hi[1]
    alpha_covered = lo[2] .<= alpha_real .<= hi[2]
    theta_covered = lo[3] .<= theta_real .<= hi[3]
    covered = [c_covered, alpha_covered, theta_covered]
    return covered
end

N_sims = 1_000
estimates = Array{Float64}(undef, N_sims, 3)
p = Progress(N_sims)
for i in 1:N_sims
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

c_real = 0.9
alpha_real = -0.7
theta_real = 3.0
# uncomment to regenerate
df = sim_mixture_cure(Weibull(exp(alpha_real), exp(theta_real)), 1 - logistic(c_real); N=10_000, thresh=100)
mcb = MixtureCureEstimator(WeibullEstimator())
mc = SurvivalModels.fit(mcb, df.t, df.observed)

lo, hi = confint(mc, df.t, df.observed)

# c_covered = lo[1] .<= c_real .<= hi[1]
# alpha_covered = lo[2] .<= alpha_real .<= hi[2]
# theta_covered = lo[3] .<= theta_real .<= hi[3]
# covered = [c_covered, alpha_covered, theta_covered]
using SurvivalModels: neg_log_likelihood
ps = [-51.487038016963815, 69.02711352578525, -388.0322041182619]
nll(x) = neg_log_likelihood(mcb, df.t, df.observed, x)
H = ForwardDiff.hessian(x -> nll(x), ps)

t = nll(ps)
d = ForwardDiff.gradient(x -> nll(x), ps)
