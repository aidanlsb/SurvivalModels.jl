module SurvivalModels

using Optim, LogExpFunctions, ForwardDiff, DiffRules, LinearAlgebra, Statistics
using Distributions: Weibull, Exponential, Normal

export WeibullEstimator, ExponentialEstimator, MixtureCureEstimator, fit, confint
include("safe_exp.jl")
include("parametric.jl")

end
