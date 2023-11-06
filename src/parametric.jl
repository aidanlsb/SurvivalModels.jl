abstract type AbstractParametricEstimator end
abstract type AbstractFittedParametricEstimator end

struct WeibullEstimator <: AbstractParametricEstimator end
struct FittedWeibullEstimator <: AbstractFittedParametricEstimator 
    α::Float64
    θ::Float64
end


struct ExponentialEstimator <: AbstractParametricEstimator end
struct FittedExponentialEstimator <: AbstractFittedParametricEstimator
    θ::Float64
end


FittedWeibullEstimator(x) = FittedWeibullEstimator(x[1], x[2])
FittedExponentialEstimator(x::AbstractArray) = FittedExponentialEstimator(x[1])
struct MixtureCureEstimator <: AbstractParametricEstimator 
    base_estimator::AbstractParametricEstimator
end

struct FittedMixtureCureEstimator <: AbstractFittedParametricEstimator
    cured_fraction::Float64
    base_estimator::AbstractFittedParametricEstimator
end

# Mappings, maybe a better way to do this
fitted_from_estimator(estimator::WeibullEstimator) = FittedWeibullEstimator
fitted_from_estimator(estimator::ExponentialEstimator) = FittedExponentialEstimator
fitted_from_estimator(estimator::MixtureCureEstimator) = FittedMixtureCureEstimator

estimator_from_fitted(fitted::FittedWeibullEstimator) = WeibullEstimator()
estimator_from_fitted(fitted::FittedExponentialEstimator) = ExponentialEstimator()
estimator_from_fitted(fitted::FittedMixtureCureEstimator) = MixtureCureEstimator(estimator_from_fitted(fitted.base_estimator))

function make_fitted(estimator::AbstractParametricEstimator, params)
    fitted_type = fitted_from_estimator(estimator)
    return fitted_type(params)
end

function make_fitted(estimator::MixtureCureEstimator, params)
    cured_fraction = params[1]
    base_params = params[2:end]
    fitted_base_estimator = make_fitted(estimator.base_estimator, base_params)
    return FittedMixtureCureEstimator(cured_fraction, fitted_base_estimator)
end

# Parameter counts, to use in link function generation and optimization initialization
num_params(estimator::WeibullEstimator) = 2
num_params(estimator::ExponentialEstimator) = 1
num_params(estimator::MixtureCureEstimator) = num_params(estimator.base_estimator) + 1


# Link functions
exp_links(estimator::AbstractParametricEstimator) = [exp for _ in 1:num_params(estimator)]
param_links(estimator::WeibullEstimator) = exp_links(estimator)
param_links(estimator::ExponentialEstimator) = exp_links(estimator)
param_links(estimator::MixtureCureEstimator) = vcat([logistic], param_links(estimator.base_estimator))


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

function cumulative_hazard(estimator::ExponentialEstimator, ts, params)
    θ = params[1]
    return ts ./ θ
end

function survival_function(estimator::AbstractParametricEstimator, ts, params)
    # this can be zero, which causes numerical errors when we take the log so add eps
    return exp.(-cumulative_hazard(estimator, ts, params)) .+ 1e-25
end

function neg_log_likelihood(estimator::AbstractParametricEstimator, ts, e, params)
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

function param_optimization(estimator::AbstractParametricEstimator, obj)
    x0 = rand(num_params(estimator))
    res = optimize(obj, x0, NelderMead())
    return Optim.minimizer(res)
end

function link_to_params(link_funcs, params)
    return [lf(p) for (lf, p) in zip(link_funcs, params)]
end

function get_linker(estimator::AbstractParametricEstimator)
    link_funcs = param_links(estimator)
    linker(x) = link_to_params(link_funcs, x)
    return linker
end


function fit(estimator::AbstractParametricEstimator, ts, e)
    linker = get_linker(estimator)
    obj(x) = neg_log_likelihood(estimator, ts, e, linker(x))
    β = param_optimization(estimator, obj)
    # β_transformed = linker(β)
    return make_fitted(estimator, β)
end

get_params(estimator::FittedWeibullEstimator) = [estimator.α, estimator.θ]
get_params(estimator::FittedExponentialEstimator) = [estimator.θ]
get_params(estimator::FittedMixtureCureEstimator) = vcat([estimator.cured_fraction], get_params(estimator.base_estimator))

function confint(fitted::AbstractFittedParametricEstimator, ts, e; confidence_level=0.95)
    β = get_params(fitted) 
    estimator = estimator_from_fitted(fitted)
    linker = get_linker(estimator)
    nll(x) = neg_log_likelihood(estimator, ts, e, linker(x))
    H = ForwardDiff.hessian(x -> nll(x), β)
    variance = inv(H)
    se = sqrt.(diag(variance))
    α = 1 - confidence_level
    z = quantile(Normal(), 1 - α/2)
    ci_width = se .* z
    lower = β .- ci_width
    upper = β .+ ci_width
    return (lower=lower, upper=upper)
end


