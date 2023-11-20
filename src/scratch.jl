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

function sim_mixture_cure_reg(; N=50_000, thresh=20)

    t = Array{Float64}(undef, N)
    x = rand(N)
    observed = Array{Bool}(undef, N)
    # α = exp.(0.5 .+ x .* β_α)

    α = exp.(-0.5 .+ x .* 0.1)
    θ = exp.(1.75 .+ x .* 0.5)
    c_ = logistic.(0.25 .+ x .* 1.0)
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
                t[i] = d > 1e-25 ? d : d + 1e-25
            end
        end
    end
    df = DataFrame(t=t, x=x[:, 1], observed=observed)
    return df
end

df = sim_mixture_cure(Weibull(0.40, 15), 0.80; N=25_000, thresh=20)
mcb = MixtureCureEstimator(WeibullEstimator())
mc = SurvivalModels.fit(mcb, df.t, df.observed)
tu = sort(df, :t).t |> unique
chb = SurvivalModels.predict_cumulative_hazard(mc, tu)
dfchb = DataFrame(chb)
dfchb.t = tu
using Gadfly
plot(dfchb, x=:t, y=:cumulative_hazard, ymin=:ci_lower, ymax=:ci_upper, Geom.line, Geom.ribbon, alpha=[0.2])

df_reg = sim_mixture_cure_reg()
ts = df_reg.t
e = df_reg.observed
X = Matrix(df_reg[:, [:x]])

mcr = SurvivalModels.fit(mcb, ts, e, X)

ch = SurvivalModels.predict_cumulative_hazard(mcr, ts, X)
dfch = DataFrame(ch)
describe(dfch)



# delta method CI scratch
# use basic first
function sim_ci_ch(c, α, θ)
    df = sim_mixture_cure(Weibull(exp(α), exp(θ)), 1 - logistic(c); N=10_000, thresh=100)
    ts = df.t
    mcb = MixtureCureEstimator(WeibullEstimator())
    mc = SurvivalModels.fit(mcb, ts, df.observed)
    t = ts[1]
    ch_actual = SurvivalModels.cumulative_hazard(mc.estimator, t, [c, α, θ])

    params = mc.params
    vcov = mc.vcov
    ts = df.t[1]
    func(x) = SurvivalModels.cumulative_hazard(mc.estimator, ts, x)
    est = func(params)
    grad = ForwardDiff.gradient(func, params)
    var = grad' * vcov * grad
    se = sqrt(var)
    upper = est + 1.96 * se
    lower = est - 1.96 * se
    covered = lower <= ch_actual <= upper
    return covered
end




N_sims = 1
logitc = 1.0
loga = -0.7
logtheta = 0.5

c = logistic(logitc)
α = exp(loga)
θ = exp(logtheta)

coverage = Array{Bool}(undef, N_sims)
actuals = Array{Float64}(undef, N_sims)
estimates = Array{Float64}(undef, N_sims)
lowers = Array{Float64}(undef, N_sims)
uppers = Array{Float64}(undef, N_sims)
# ll_est = Array{Float64}(undef, N_sims)

# ll = pyimport("lifelines")
# mcp = ll.MixtureCureFitter(base_fitter=ll.WeibullFitter())

function chaz_raw(estimator, t, params_raw)
    params = SurvivalModels.transform_params(estimator, params_raw)
    return SurvivalModels.cumulative_hazard(estimator, t, params)
end

function log_chaz_raw(estimator, t, params_raw)
    params = SurvivalModels.transform_params(estimator, params_raw)
    return log(SurvivalModels.cumulative_hazard(estimator, t, params))
end

#TODOs: think about the log transform approach to avoid negatives in the cumulative hazard
# think about whether need to define more wrappers around CH etc (probably not, just deal with in CI methods)
@showprogress for i in 1:N_sims
    N = 25_000
    df = sim_mixture_cure(Weibull(α, θ), 1 - c; N=N, thresh=100)
    mcb = MixtureCureEstimator(WeibullEstimator())
    mc = SurvivalModels.fit(mcb, df.t, df.observed)
    # mcp.fit(df.t, event_observed=df.observed)

    teval = 1.0
    params_raw = SurvivalModels.fitted_params(mc)
    ch_actual = chaz_raw(mcb, teval, [logitc, loga, logtheta])
    actuals[i] = ch_actual
    # params = SurvivalModels.transform_params(mcb, SurvivalModels.fitted_params(mc))
    # print(params)
    vcov = mc.vcov
    func(x) = log_chaz_raw(mc.estimator, teval, x)

    # try
    est = exp(func(params_raw))
    estimates[i] = est
    # catch
    # println("t is $(teval)")
    # println("params are $(params)")
    # end
    grad = ForwardDiff.gradient(func, params_raw)
    var = grad' * vcov * grad
    se = sqrt(var)
    # changing these
    upper = est * exp(1.96 * se)
    lower = est * exp(-1.96 * se)
    uppers[i] = upper
    lowers[i] = lower
    covered = lower <= ch_actual <= upper
    coverage[i] = covered

end

df_sim = DataFrame(actuals=actuals, estimates=estimates, lower=lowers, upper=uppers, covered=coverage)

print("Coverage for cumulative_hazard is: $(mean(coverage))")




N = 25_000
df = sim_mixture_cure(Weibull(α, θ), 1 - c; N=N, thresh=100)
mcb = MixtureCureEstimator(WeibullEstimator())
mc = SurvivalModels.fit(mcb, df.t, df.observed)
tpred = sort(df, :t).t |> unique
ch = DataFrame(predict_cumulative_hazard(mc, tpred))
ch.t = tpred

using Gadfly
plot(ch, x=:t, y=:cumulative_hazard, ymin=:ci_lower, ymax=:ci_upper, Geom.line, Geom.ribbon, alpha=[0.2])




function compare_lifelines(df, mc)
    ll = pyimport("lifelines")
    mcpy = ll.MixtureCureFitter(base_fitter=ll.WeibullFitter())
    mcpy.fit(df.t, event_observed=df.observed)

    println("Cured fraction from lifelines is $(1 - mcpy.cured_fraction_) and from this package is $(logistic(mc.params[1]))")
    println("α from lifelines is $(mcpy.rho_) and from this package is $(exp(mc.params[2]))")
    println("θ from lifelines is $(mcpy.lambda_) and from this package is $(exp(mc.params[3]))")
    return nothing
end

function run_sim()
    c_real = 0.9
    alpha_real = -0.7
    theta_real = 3.0
    df = sim_mixture_cure(Weibull(exp(alpha_real), exp(theta_real)), 1 - logistic(c_real); N=10_000, thresh=100)
    mcb = MixtureCureEstimator(WeibullEstimator())
    mc = SurvivalModels.fit(mcb, df.t, df.observed)

    lo, hi = confint(mc)

    c_covered = lo[1] .<= c_real .<= hi[1]
    alpha_covered = lo[2] .<= alpha_real .<= hi[2]
    theta_covered = lo[3] .<= theta_real .<= hi[3]
    covered = [c_covered, alpha_covered, theta_covered]
    return covered
end

function check_cis()
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
    return nothing
end




