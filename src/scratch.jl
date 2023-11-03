using SurvivalModels, Distributions, DataFramesMeta, PythonCall, ForwardDiff, DiffRules

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

df = sim_mixture_cure(Weibull(0.5, 120), 0.3; N=100_000, thresh=100)
mc = SurvivalModels.fit(MixtureCureEstimator(WeibullEstimator()), df.x, df.observed)

ll = pyimport("lifelines")
mcpy = ll.MixtureCureFitter(base_fitter=ll.WeibullFitter())
mcpy.fit(df.x, event_observed=df.observed)

println("Cured fraction from lifelines is $(1 - mcpy.cured_fraction_) and from this package is $(mc.cured_fraction)")
println("α from lifelines is $(mcpy.rho_) and from this package is $(mc.base_estimator.α)")
println("θ from lifelines is $(mcpy.lambda_) and from this package is $(mc.base_estimator.θ)")



