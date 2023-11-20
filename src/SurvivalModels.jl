module SurvivalModels

using Optim, LogExpFunctions, ForwardDiff, DiffRules, LinearAlgebra, Statistics, DataFrames
using Distributions: Weibull, Exponential, Normal

export WeibullEstimator, ExponentialEstimator, MixtureCureEstimator, RidgePenaltyEstimator, fit, confint, results_summary, predict_cumulative_hazard, coef
include("safe_exp.jl")
include("parametric.jl")

end
