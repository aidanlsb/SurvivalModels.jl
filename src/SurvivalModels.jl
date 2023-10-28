module SurvivalModels

using Optim, LogExpFunctions

export WeibullEstimator, ExponentialEstimator, MixtureCureEstimator, fit
include("parametric.jl")

end
