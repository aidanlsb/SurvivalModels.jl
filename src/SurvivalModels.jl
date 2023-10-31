module SurvivalModels

using Optim, LogExpFunctions, Optimization, OptimizationOptimJL

export WeibullEstimator, ExponentialEstimator, MixtureCureEstimator, fit
include("parametric.jl")

end
