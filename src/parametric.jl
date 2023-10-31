abstract type ParametricEstimator end
abstract type FittedParametricEstimator end

struct WeibullEstimator <: ParametricEstimator end
struct FittedWeibullEstimator <: FittedParametricEstimator 
    α::Float64
    θ::Float64
end

struct ExponentialEstimator <: ParametricEstimator end
struct FittedExponentialEstimator <: FittedParametricEstimator
    θ::Float64
end

FittedWeibullEstimator(x) = FittedWeibullEstimator(x[1], x[2])
FittedExponentialEstimator(x::AbstractArray) = FittedExponentialEstimator(x[1])

struct MixtureCureEstimator <: ParametricEstimator 
    base_estimator::ParametricEstimator
end

struct FittedMixtureCureEstimator <: FittedParametricEstimator
    cured_fraction::Float64
    base_estimator::FittedParametricEstimator
end

# Mappings, maybe a better way to do this
fitted_from_estimator(estimator::WeibullEstimator) = FittedWeibullEstimator
fitted_from_estimator(estimator::ExponentialEstimator) = FittedExponentialEstimator

function make_fitted(estimator::ParametricEstimator, params)
    fitted_type = fitted_from_estimator(estimator)
    return fitted_type(params)
end

function make_fitted(estimator::MixtureCureEstimator, params)
    cured_fraction = params[1]
    fitted_base_estimator = make_fitted(estimator.base_estimator, params[2:end])
    return FittedMixtureCureEstimator(cured_fraction, fitted_base_estimator)
end

# Parameter counts, to use in link function generation and optimization initialization
num_params(estimator::WeibullEstimator) = 2
num_params(estimator::ExponentialEstimator) = 1
num_params(estimator::MixtureCureEstimator) = num_params(estimator.base_estimator) + 1

# Link functions
exp_links(estimator::ParametricEstimator) = [exp for _ in 1:num_params(estimator)]
param_links(estimator::WeibullEstimator) = exp_links(estimator)
param_links(estimator::ExponentialEstimator) = exp_links(estimator)
param_links(estimator::MixtureCureEstimator) = vcat([logistic], param_links(estimator.base_estimator))

# TODO: figure out what to do here
const MAX = floatmax(Float64)
function safe_exp(x)
    return clamp.(exp.(x), -Inf, MAX)
end
# function safe_exp(x)
#     return exp.(x)
# end

function log_hazard(estimator::WeibullEstimator, ts, params)
    α = params[1]
    θ = params[2]
    return log(α) .- log(θ) .+ (α - 1) .* (log.(ts) .- log(θ))
end

function cumulative_hazard(estimator::WeibullEstimator, ts, params)
    α = params[1]
    θ = params[2]
    return safe_exp.(α .* (log.(ts) .- log(θ)))
end


function log_hazard(estimator::ExponentialEstimator, ts, params)
    θ = params[1]
    return -log(θ)
end

function cumulative_hazard(esimator::ExponentialEstimator, ts, params)
    θ = params[1]
    return ts ./ θ
end

function survival_function(estimator::ParametricEstimator, ts, params)
    # this can be zero, which causes numerical errors when we take the log so add small positive value
    return exp.(-cumulative_hazard(estimator, ts, params)) .+ 1e-25
end

function neg_log_likelihood(estimator::ParametricEstimator, ts, e, params)
    lh = e .* log_hazard(estimator, ts, params)
    ch = cumulative_hazard(estimator, ts, params)
    ll = sum(lh) - sum(ch)
    return -1.0 * ll
end

# Mixture Cure
function log_hazard(estimator::MixtureCureEstimator, ts, params)
    c = params[1]
    base_params = params[2:end]
    s = survival_function(estimator.base_estimator, ts, base_params)
    lh = log_hazard(estimator.base_estimator, ts, base_params)
    ch = -1.0 .* log.((1.0 - c) .+ c .* s)
    return log(c) .+ log.(s) .+ lh .+ ch
end

function cumulative_hazard(estimator::MixtureCureEstimator, ts, params)
    c = params[1]
    base_params = params[2:end]
    s = survival_function(estimator.base_estimator, ts, base_params)
    return -1.0 .* log.(c .* s .+ (1 .- c))
end

function neg_log_likelihood(estimator::MixtureCureEstimator, ts, e, params)
    lh = e .* log_hazard(estimator, ts, params)
    ch = cumulative_hazard(estimator, ts,  params)
    ll = sum(lh) - sum(ch)
    return -1.0 * ll
end

# function param_optimization(estimator::ParametricEstimator, obj)
#     x0 = rand(num_params(estimator))
#     res = optimize(obj, x0, NelderMead())
#     return Optim.minimizer(res)
# end

# function param_optimization(estimator::ParametricEstimator, obj, p)
#     obj(x, p) = neg_log_likelihood(estimator, p[:, 1], p[:, 2], )

#     optf = OptimizationFunction(obj_mc_2, Optimization.AutoForwardDiff())
#     prob = OptimizationProblem(optf, x0_mc, p)
#     u = solve(prob, LBFGS())
#     return u


function link_to_params(link_funcs, params)
    return [lf(p) for (lf, p) in zip(link_funcs, params)]
end

function fit(estimator::ParametricEstimator, ts, e)
    link_funcs = param_links(estimator)
    linker(x) = link_to_params(link_funcs, x)

    # new
    p = [ts e]
    obj(x, p) = neg_log_likelihood(estimator, p[:, 1], p[:, 2], linker(x))
    optf = OptimizationFunction(obj, Optimization.AutoForwardDiff())

    x0 = randn(num_params(estimator))
    prob = OptimizationProblem(optf, x0, p)
    x = solve(prob, NelderMead())
    print(x.original)
    # x = param_optimization(estimator, obj)
    x_transformed = linker(x)
    return make_fitted(estimator, x_transformed)
end