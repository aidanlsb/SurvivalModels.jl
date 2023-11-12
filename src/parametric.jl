abstract type AbstractParametricEstimator end
abstract type AbstractFittedParametricEstimator end

struct WeibullEstimator <: AbstractParametricEstimator end
struct FittedWeibullEstimator <: AbstractFittedParametricEstimator 
    α::Float64
    θ::Float64
end

# TODO: remove super type if unnecessary
struct FittedParametricEstimator{T} <: AbstractFittedParametricEstimator where T <: Real
    estimator::AbstractParametricEstimator
    params::Array{T}
end


struct ExponentialEstimator <: AbstractParametricEstimator end
struct MixtureCureEstimator <: AbstractParametricEstimator 
    base_estimator::AbstractParametricEstimator
end

estimator_from_fitted(fitted::FittedParametricEstimator) = fitted.estimator

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
    return log(α) - log(θ) + (α - 1) * (log(ts) - log(θ))
end


function cumulative_hazard(estimator::WeibullEstimator, ts, params)
    α = params[1]
    θ = params[2]
    return safe_exp(α * (log(ts) - log(θ)))
end

function log_hazard(estimator::ExponentialEstimator, ts, params)
    θ = params[1]
    return -log(θ)
end

function cumulative_hazard(estimator::ExponentialEstimator, ts, params)
    θ = params[1]
    return ts / θ
end

# Mixture Cure
function log_hazard(estimator::MixtureCureEstimator, ts, params)
    c = params[1]
    base_params = params[2:end]
    s = survival_function(estimator.base_estimator, ts, base_params)
    lh = log_hazard(estimator.base_estimator, ts, base_params)
    ch = -1.0 * log((1.0 - c) + c * s)
    return log(c) + log(s) + lh + ch
end

function cumulative_hazard(estimator::MixtureCureEstimator, ts, params)
    c = params[1]
    base_params = params[2:end]
    s = survival_function(estimator.base_estimator, ts, base_params)
    return -1.0 * log(c * s + (1 - c))
end


function survival_function(estimator::AbstractParametricEstimator, ts, params)
    # this can be zero, which causes numerical errors when we take the log so add eps
    # TODO: maybe can remove `safe`?
    return safe_exp.(-cumulative_hazard(estimator, ts, params)) + 1e-25
end

function link_to_params(link_funcs, params::Vector)
    return [lf(p) for (lf, p) in zip(link_funcs, params)]
end

function link_to_params(link_funcs, params::Matrix)
    n, m = size(params)
    output = Array{eltype(params)}(undef, n, m)
    for i in 1:n
        output[i, :] = link_to_params(link_funcs, params[i, :])
    end
    return output
end

"""
Compute negative log likelihood for a single observation and set of parameters.
"""
function neg_log_likelihood_one(estimator::AbstractParametricEstimator, ts::T, e::Bool, transformed_params::Vector) where T <: Real
    lh = e * log_hazard(estimator, ts, transformed_params)
    # println(lh)
    ch = cumulative_hazard(estimator, ts, transformed_params)
    # println(ch)
    return -1.0 * (lh - ch)
end

"""
Compute NLL for all observations, assuming one set of parameters (i.e., not the regression case).
"""
function neg_log_likelihood_inner(estimator::AbstractParametricEstimator, ts::Vector{T}, e::Vector{Bool}, transformed_params::Vector) where T <: Real
    ll = 0.0
    # ll = Float64[]
    N = length(ts)
    for i in 1:N
        ll += neg_log_likelihood_one(estimator, ts[i], e[i], transformed_params)
        # push!(ll, lli)
    end
    return ll
end

"""
Compute NLL for all observations, assuming individual params for each obs (i.e., the regression case).
"""
function neg_log_likelihood_inner(estimator::AbstractParametricEstimator, ts::Vector{T}, e::Vector{Bool}, transformed_params::Matrix) where T <: Real
    ll = 0.0
    # ll = Float64[]
    N = length(ts)
    for i in 1:N
        ll += neg_log_likelihood_one(estimator, ts[i], e[i], transformed_params[i, :])
        # push!(ll, lli)
    end
    return ll
end

function transform_params(estimator::AbstractParametricEstimator, params)
    link_funcs = param_links(estimator)
    transformed_params = link_to_params(link_funcs, params)
    return transformed_params
end

function neg_log_likelihood(estimator::AbstractParametricEstimator, ts, e, params)
    transformed_params = transform_params(estimator, params)
    return neg_log_likelihood_inner(estimator, ts, e, transformed_params)
end

""" Given a vector of initialized params and X, deal with the shapes and \beta in matrix form."""
function reshape_params(estimator::AbstractParametricEstimator, X, β)
    num_parameters = num_params(estimator)
    # add one for the intercept - maybe need to revisit this
    num_predictors = size(X, 2) + 1
    return transpose(reshape(β, num_parameters, num_predictors))
end

function compute_params(estimator::AbstractParametricEstimator, X, params)
    β = reshape_params(estimator, X, params)
    # add an intercept
    nobs = size(X, 1)
    Xb = hcat(ones(nobs), X)
    return Xb * β
end

function neg_log_likelihood(estimator::AbstractParametricEstimator, ts, e, X, params)
    Xb = compute_params(estimator, X, params)
    transformed_params = transform_params(estimator, Xb)
    return neg_log_likelihood_inner(estimator, ts, e, transformed_params)
end

function param_optimization(estimator::AbstractParametricEstimator, obj, x0)
    func = TwiceDifferentiable(obj, x0, autodiff=:forward)
    res = optimize(func, x0, LBFGS())
    # println(res)
    return Optim.minimizer(res)
end

# TODO: need to define methods for other estimators, use them here
function initialize_params(estimator::WeibullEstimator, ts, e)
    x0 = zeros(num_params(estimator))
    x0[end] = log(mean(ts))
    return x0
end

function initialize_params(estimator::WeibullEstimator, ts, e, X)
    num_predictors = size(X, 2)
    num_parameters = num_params(estimator)
end

function initialize_params(estimator::MixtureCureEstimator, ts, e)
    x0_base = initialize_params(estimator.base_estimator, ts, e)
    x0_c = logit(mean(e))
    return vcat(x0_c, x0_base)
end

# TODO: improve this, need to reference base estimator
function initialize_params(estimator::MixtureCureEstimator, ts, e, X)
    num_predictors = size(X, 2)
    num_parameters = num_params(estimator)
    intercepts = zeros(num_parameters)
    intercepts[1] = logit(mean(e))
    intercepts[end] = log(mean(ts))
    β = zeros(num_predictors * (num_parameters))
    return vcat(intercepts, β)
end

function fit(estimator::AbstractParametricEstimator, ts, e)
    obj(β) = neg_log_likelihood(estimator, ts, e, β)
    β0 = initialize_params(estimator, ts, e)
    β = param_optimization(estimator, obj, β0)
    return FittedParametricEstimator(estimator, β)
end


function fit(estimator::AbstractParametricEstimator, ts, e, X)
    obj(β) = neg_log_likelihood(estimator, ts, e, X, β)
    β0 = initialize_params(estimator, ts, e, X)
    β = param_optimization(estimator, obj, β0)
    return FittedParametricEstimator(estimator, β)
end

get_params(estimator::FittedParametricEstimator) = estimator.params

function confint(fitted::FittedParametricEstimator, ts, e; confidence_level=0.95)
    β = get_params(fitted) 
    estimator = estimator_from_fitted(fitted)
    nll(x) = neg_log_likelihood(estimator, ts, e, x)
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

function confint(fitted::FittedParametricEstimator, ts, e, X; confidence_level=0.95)
end


