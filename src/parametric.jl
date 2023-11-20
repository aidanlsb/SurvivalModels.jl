abstract type AbstractParametricEstimator end
abstract type AbstractFittedParametricEstimator end

struct WeibullEstimator <: AbstractParametricEstimator end

struct FittedParametricEstimator <: AbstractParametricEstimator
    estimator::AbstractParametricEstimator
    params::Array{Float64}
    vcov::Matrix{Float64}
    stderrors::Union{Array{Float64}, Nothing}
    param_names::Array{String}
    optim_result::Optim.MultivariateOptimizationResults
    fitted_with_covariates::Bool
end


struct ExponentialEstimator <: AbstractParametricEstimator end
struct MixtureCureEstimator{T} <: AbstractParametricEstimator where T <: AbstractParametricEstimator
    base_estimator::T
end

# struct RidgePenaltyEstimator{T} <: AbstractParametricEstimator where T <: AbstractParametricEstimator
#     estimator::T
#     λ::Float64
# end

# Parameter counts, to use in optimization initialization
num_params(estimator::WeibullEstimator) = 2
num_params(estimator::ExponentialEstimator) = 1
num_params(estimator::MixtureCureEstimator) = num_params(estimator.base_estimator) + 1
# num_params(estimator::RidgePenaltyEstimator) = num_params(estimator.estimator)

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

function survival_function(estimator::AbstractParametricEstimator, ts, params)
    # this can become zero, which causes numerical errors when we take the log so add eps
    return safe_exp.(-cumulative_hazard(estimator, ts, params)) + 1e-25
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


# Ridge
# function log_hazard(estimator::RidgePenaltyEstimator, ts, params)
#     return log_hazard(estimator.estimator, ts, params)
# end
# function cumulative_hazard(estimator::RidgePenaltyEstimator, ts, params)
#     return cumulative_hazard(estimator.estimator, ts, params)
# end


function link_to_params(estimator::Union{ExponentialEstimator, WeibullEstimator}, params::Vector{T}) where T <: Real
    return exp.(params)
    # return log1pexp.(params)
end

function link_to_params(estimator::MixtureCureEstimator, params::Vector{T}) where T <: Real
    return vcat(logistic(params[1]), link_to_params(estimator.base_estimator, params[2:end]))
end

function link_to_params(estimator::AbstractParametricEstimator, params::Matrix{T}) where T <: Real
    output = similar(params)
    N = size(params, 1)
    for i in 1:N
        output[i, :] = link_to_params(estimator, params[i, :])
    end
    return output
end



function create_intercept(X)
    nobs = size(X, 1)
    intercept = ones(eltype(X), nobs)
    return hcat(intercept, X)
end

function initialize_params(estimator::Union{ExponentialEstimator, WeibullEstimator}, ts, e)
    β0 = zeros(Float64, num_params(estimator))
    β0[end] = log(mean(ts))
    # β0[end] = logexpm1(mean(ts))
    return β0
end


function initialize_params(estimator::Union{ExponentialEstimator, WeibullEstimator}, ts, e, X)
    num_predictors = size(X, 2)
    num_parameters = num_params(estimator)
    β0 = zeros(Float64, num_predictors * num_parameters)  
    # first coefs are the intercepts for each parameter
    intercepts = @view β0[1:num_parameters]
    # for the weibull, initialize the theta parameter to the log mean of durations
    intercepts[end] = log(mean(ts))
    # intercepts[end] = logexpm1(mean(ts))
    return β0
end

function initialize_params(estimator::MixtureCureEstimator, ts, e)
    β0_base = initialize_params(estimator.base_estimator, ts, e)
    β0_c = logit(mean(e))
    return vcat(β0_c, β0_base)
end

function initialize_params(estimator::MixtureCureEstimator, ts, e, X)
    num_params_base = num_params(estimator.base_estimator)
    num_predictors = size(X, 2)
    β0_base = initialize_params(estimator.base_estimator, ts, e, X)
    intercepts_base = β0_base[1:num_params_base]
    coefs_base = β0_base[num_params_base+1:end]
    β0_c = zeros(num_predictors)
    β0_c[1] = logit(mean(e))
    return vcat(β0_c[1], intercepts_base, β0_c[2:end], coefs_base)
end

""" Given a vector of initialized params and X, deal with the shapes and \beta in matrix form."""
function reshape_params(estimator::AbstractParametricEstimator, X, β)
    num_parameters = num_params(estimator)
    num_predictors = size(X, 2)
    return transpose(reshape(β, num_parameters, num_predictors))
end

""" Given a vector of coefficients and covariates, output the linked parameters."""
function coefs_to_params(estimator::AbstractParametricEstimator, X, coefs)
    β = reshape_params(estimator, X, coefs)
    Xb = X * β
    return link_to_params(estimator, Xb)
end

""" In the non-regression case, just apply the link functions."""
function coefs_to_params(estimator::AbstractParametricEstimator, coefs)
    return link_to_params(estimator, coefs)
end

"""
Compute negative log likelihood for a single observation and set of parameters.
"""
#TODO: improve type annotation here
function neg_log_likelihood_one(estimator::AbstractParametricEstimator, ts::T, e::Bool, transformed_params::Vector) where T <: Real
    lh = e * log_hazard(estimator, ts, transformed_params)
    ch = cumulative_hazard(estimator, ts, transformed_params)
    return -1.0 * (lh - ch)
end


"""
Compute NLL for all observations, assuming one set of parameters (i.e., not the regression case).
"""
function neg_log_likelihood_inner(estimator::AbstractParametricEstimator, ts::Vector{T}, e::Union{Vector{Bool}, BitVector}, params::Vector) where T <: Real
    ll = 0.0
    N = length(ts)
    @inbounds for i in 1:N
        ll += neg_log_likelihood_one(estimator, ts[i], e[i], params)
    end
    return ll
end

"""
Compute NLL for all observations, assuming individual params for each obs (i.e., the regression case).
"""
function neg_log_likelihood_inner(estimator::AbstractParametricEstimator, ts::Vector{T}, e::Union{Vector{Bool}, BitVector}, params::Matrix) where T <: Real
    ll = 0.0
    N = length(ts)
    @inbounds for i in 1:N
        ll += neg_log_likelihood_one(estimator, ts[i], e[i], params[i, :])
    end
    return ll
end

function neg_log_likelihood(estimator::AbstractParametricEstimator, ts, e, coefs)
    params = coefs_to_params(estimator, coefs)
    return neg_log_likelihood_inner(estimator, ts, e, params)
end

# function neg_log_likelihood(estimator::RidgePenaltyEstimator, ts, e, params)
#     num_parameters = num_params(estimator.estimator)
#     params_to_penalize = params[(num_parameters+1):end]
#     penalty = sum(params_to_penalize .^ 2) * estimator.λ
#     return neg_log_likelihood(estimator.estimator, ts, e, transformed_params) + penalty
# end


function neg_log_likelihood(estimator::AbstractParametricEstimator, ts, e, X, coefs)
    params = coefs_to_params(estimator, X, coefs)
    return neg_log_likelihood_inner(estimator, ts, e, params)
end

function param_optimization(obj, x0)
    func = TwiceDifferentiable(obj, x0, autodiff=:forward)
    res = optimize(func, x0, LBFGS())
    return res
end

function calculate_variance(nll, β)
    H = ForwardDiff.hessian(x -> nll(x), β)
    variance = inv(H)
    return variance
end

function calculate_stderrors(variance)
    ses = sqrt.(diag(variance))
    return ses
end

param_names(estimator::WeibullEstimator) = ["α", "θ"]
param_names(estimator::MixtureCureEstimator) = vcat("c", param_names(estimator.base_estimator))
# param_names(estimator::RidgePenaltyEstimator) = param_names(estimator.estimator)

function param_names(estimator::AbstractParametricEstimator, X::Matrix{T}) where T <: Real
    num_predictors = size(X, 2)
    param_names_base = param_names(estimator)
    param_names_all = String[]
    for i in 1:num_predictors
        for p in param_names_base
            push!(param_names_all, "$(p)_$(i)")
        end
    end
    return param_names_all
end

function fit(estimator::AbstractParametricEstimator, ts, e)
    nll(β) = neg_log_likelihood(estimator, ts, e, β)
    β0 = initialize_params(estimator, ts, e)
    res = param_optimization(nll, β0)
    β = Optim.minimizer(res)
    vcov = calculate_variance(nll, β)
    stderrors = calculate_stderrors(vcov)
    names = param_names(estimator)
    return FittedParametricEstimator(estimator, β, vcov, stderrors, names, res, false)
end

function fit(estimator::AbstractParametricEstimator, ts, e, X; add_intercept=true)
    if add_intercept
        X_input = create_intercept(X)
    else
        X_input = X
    end

    nll(β) = neg_log_likelihood(estimator, ts, e, X_input, β)
    β0 = initialize_params(estimator, ts, e, X_input)
    res = param_optimization(nll, β0)
    β = Optim.minimizer(res)
    vcov = calculate_variance(nll, β)
    stderrors = calculate_stderrors(vcov)
    names = param_names(estimator, X_input)
    return FittedParametricEstimator(estimator, β, vcov, stderrors, names, res, true)
end

# function fit(estimator::RidgePenaltyEstimator, ts, e, X; add_intercept=true)
#     if add_intercept
#         X_input = create_intercept(X)
#     else
#         X_input = X
#     end

#     nll(β) = neg_log_likelihood(estimator, ts, e, X_input, β)
#     β0 = initialize_params(estimator, ts, e, X_input)
#     res = param_optimization(estimator, nll, β0)
#     β = Optim.minimizer(res)
#     names = param_names(estimator, X_input)
#     return FittedParametricEstimator(estimator, β, nothing, names, res)
# end

coef(estimator::FittedParametricEstimator) = estimator.params
stderror(estimator::FittedParametricEstimator) = estimator.stderrors
param_names(fitted::FittedParametricEstimator) = fitted.param_names


function zvalue(confidence_level)
    α = 1 - confidence_level
    return quantile(Normal(), 1 - α/2)
end

function confint(fitted::FittedParametricEstimator; confidence_level=0.95)
    β = coef(fitted)
    se = stderror(fitted)
    z = zvalue(confidence_level)
    ci_width = se .* z
    lower = β .- ci_width
    upper = β .+ ci_width
    return (lower=lower, upper=upper)
end

function results_summary(fitted::FittedParametricEstimator; confidence_level=0.95)
    coefs = coef(fitted)
    cis = confint(fitted; confidence_level)
    names = param_names(fitted)
    return DataFrame(parameter=names, coef=coefs, ci_lower=cis.lower, ci_upper=cis.upper) 
end

function check_fit_type(fitted::FittedParametricEstimator, needs_covariates)
    if needs_covariates & !fitted.fitted_with_covariates
        throw(ArgumentError("This estimator was not fit with coveriates. Call `fit` only on a vector of durations to obtain predictions, or refit a model with covariates."))
    elseif !needs_covariates & fitted.fitted_with_covariates
        throw(ArgumentError("This estimator was fit with covariates. Pass a matrix of covariate values in `fit` to obtain predictions, or refit a model without covariates."))
    end
end

# TODO: rename this
# function fitted_params(fitted::FittedParametricEstimator, X; add_intercept=true)
#     if add_intercept
#         X_input = create_intercept(X)
#     else
#         X_input = X
#     end
#     coefs = coef(fitted)
#     coefs_reshaped = reshape_params(fitted.estimator, X_input, coefs)
#     params = X_input * coefs_reshaped
#     return params
# end


function cumulative_hazard_from_coefs(estimator::AbstractParametricEstimator, ts, coefs)
    params = coefs_to_params(estimator, coefs)
    return cumulative_hazard(estimator, ts, params)
end

function cumulative_hazard_from_coefs(estimator::AbstractParametricEstimator, ts, X, coefs)
    params = coefs_to_params(estimator, X, coefs)[1, :]
    return cumulative_hazard(estimator, ts, params)
end

function log_cumulative_hazard_from_coefs(estimator::AbstractParametricEstimator, ts, coefs)
    chaz = cumulative_hazard_from_coefs(estimator, ts, coefs)
    return log(chaz)
end

function log_cumulative_hazard_from_coefs(estimator::AbstractParametricEstimator, ts, X, coefs)
    chaz = cumulative_hazard_from_coefs(estimator, ts, X, coefs)
    return log(chaz)
end

# add non-regression method

function predict_cumulative_hazard(fitted, ts; confidence_level=0.95)
    check_fit_type(fitted, false)
    coefs = coef(fitted)
    nt = length(ts)
    chs = Array{Float64}(undef, nt)
    lowers = Array{Float64}(undef, nt)
    uppers = Array{Float64}(undef, nt)
    vcov = fitted.vcov
    z = zvalue(confidence_level)
    for i in 1:nt
        ch = cumulative_hazard_from_coefs(fitted.estimator, ts[i], coefs)
        chs[i] = ch
        grad = ForwardDiff.gradient(c -> log_cumulative_hazard_from_coefs(fitted.estimator, ts[i], c), coefs)
        var = grad' * vcov * grad
        se = sqrt(var)
        lower = ch * exp(-z * se)
        upper = ch * exp(z * se)
        lowers[i] = lower
        uppers[i] = upper
    end
    return (cumulative_hazard=chs, ci_lower=lowers, ci_upper=uppers)
end

function predict_cumulative_hazard(fitted, ts, X; add_intercept=true, confidence_level=0.95)
    check_fit_type(fitted, true)
    if add_intercept
        X_input = create_intercept(X)
    else
        X_input = X
    end
    coefs = coef(fitted)
    nt = length(ts)
    chs = Array{Float64}(undef, nt)
    lowers = Array{Float64}(undef, nt)
    uppers = Array{Float64}(undef, nt)
    vcov = fitted.vcov
    z = zvalue(confidence_level)
    for i in 1:nt
        # keep as matrix
        Xi = X_input[i:i, :]
        ch = cumulative_hazard_from_coefs(fitted.estimator, ts[i], Xi, coefs)
        chs[i] = ch
        grad = ForwardDiff.gradient(c -> log_cumulative_hazard_from_coefs(fitted.estimator, ts[i], Xi, c), coefs)
        var = grad' * vcov * grad
        se = sqrt(var)
        lower = ch * exp(-z * se)
        upper = ch * exp(z * se)
        lowers[i] = lower
        uppers[i] = upper
    end
    return (cumulative_hazard=chs, ci_lower=lowers, ci_upper=uppers)
end
