
# exp that is capped with some room
const MAX = log(floatmax(Float64)) - 75
function safe_exp(x)
    return exp(clamp(x, -Inf, MAX))
end

# custom rule for autodiff
function define_∂safe_exp()
    ∂safe_exp(x) = safe_exp(x)
    DiffRules.@define_diffrule SurvivalModels.safe_exp(x) = :(∂safe_exp($x))
    eval(ForwardDiff.unary_dual_definition(:SurvivalModels, :safe_exp))
end