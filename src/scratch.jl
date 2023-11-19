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
    t = Array{Float64}(undef, N)
    x = rand(Normal(), (N, 2))
    observed = Array{Bool}(undef, N)
    # α = exp.(0.5 .+ x .* β_α)
    α = exp.(0.5 .+ x[:, 1] .* β_α .+ β_α*0.5 .* x[:, 2])
    θ = exp.(1.75 .+ x[:, 1] .* β_θ .+ β_θ * 0.5 .* x[:, 2])
    c_ = logistic.(2.0 .+ x[:, 1] .* β_c .+ β_c * 0.5 .* x[:, 2])
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
    df = DataFrame(t=t, x=x[:, 1], x2=x[:, 2], observed=observed)
    df.intercept = ones(Float64, nrow(df))
    return df
end

df = sim_mixture_cure(Weibull(0.40, 130945), 0.80; N=25_000, thresh=30_000)
mcb = MixtureCureEstimator(WeibullEstimator())
mc = SurvivalModels.fit(mcb, df.t, df.observed)
p = predict_cumulative_hazard(mc, df.t)

df_reg = sim_mixture_cure_reg([2.3], [1.5], [0.4]; N=25_000, thresh=10)
t = df_reg.t
e = df_reg.observed
X = Matrix(df_reg[:, [:x, :x2]])

mcr = SurvivalModels.fit(mcb, t, e, X)
results_summary(mcr)

preg = predict_cumulative_hazard(mcr, t, X)
println(sum(preg))
println(sum(e))

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




N_sims = 100
logitc = 1.0
loga = -0.7
logtheta = 3.0

c = logistic(logitc)
α = exp(loga)
θ = exp(logtheta)

coverage = Array{Bool}(undef, N_sims)
actuals = Array{Float64}(undef, N_sims)
estimates = Array{Float64}(undef, N_sims)
lowers = Array{Float64}(undef, N_sims)
uppers = Array{Float64}(undef, N_sims)
# ll_est = Array{Float64}(undef, N_sims)

ll = pyimport("lifelines")
mcp = ll.MixtureCureFitter(base_fitter=ll.WeibullFitter())

function chaz_raw(estimator, t, params_raw)
    params = SurvivalModels.transform_params(estimator, params_raw)
    return SurvivalModels.cumulative_hazard(estimator, t, params)
end

@showprogress for i in 1:N_sims
    N = 25_000
    df = sim_mixture_cure(Weibull(α, θ), 1 - c; N=N, thresh=100)
    mcb = MixtureCureEstimator(WeibullEstimator())
    mc = SurvivalModels.fit(mcb, df.t, df.observed)
    mcp.fit(df.t, event_observed=df.observed)

    teval = 1.0
    params_raw = SurvivalModels.fitted_params(mc)
    ch_actual = chaz_raw(mcb, teval, [logitc, loga, logtheta])
    actuals[i] = ch_actual
    # params = SurvivalModels.transform_params(mcb, SurvivalModels.fitted_params(mc))
    # print(params)
    vcov = mc.vcov
    func(x) = chaz_raw(mc.estimator, teval, x)

    try
        est = func(params_raw)
        estimates[i] = est
    catch
        println("t is $(teval)")
        println("params are $(params)")
    end
    grad = ForwardDiff.gradient(func, params_raw)
    var = grad' * vcov * grad
    se = sqrt(var)
    upper = est + 1.96 * se
    lower = est - 1.96 * se
    uppers[i] = upper
    lowers[i] = lower
    covered = lower <= ch_actual <= upper
    coverage[i] = covered

end

df_sim = DataFrame(actuals=actuals, estimates=estimates, lower=lowers, upper=uppers, covered=coverage)

print("Coverage for cumulative_hazard is: $(mean(coverage))")







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




