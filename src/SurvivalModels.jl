module SurvivalModels

using Optim, LogExpFunctions, Optimization, OptimizationOptimJL, ForwardDiff, DiffRules 

export WeibullEstimator, ExponentialEstimator, MixtureCureEstimator, fit
include("safe_exp.jl")
include("parametric.jl")

end
