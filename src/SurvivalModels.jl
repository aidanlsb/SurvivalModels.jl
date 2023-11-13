module SurvivalModels

using Optim, LogExpFunctions, ForwardDiff, DiffRules, LinearAlgebra, Statistics, DataFrames
using Distributions: Weibull, Exponential, Normal

export WeibullEstimator, ExponentialEstimator, MixtureCureEstimator, fit, confint, results_summary
include("safe_exp.jl")
include("parametric.jl")

end
