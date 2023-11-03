
# TODO: need to figure out the proper treatment of the derivative
const MAX = log(floatmax(Float64)) - 75
function safe_exp(x)
    return exp(clamp(x, -Inf, MAX))
end

# custom rule for autodiff
∂safe_exp(x) = safe_exp(x)
DiffRules.@define_diffrule SurvivalModels.safe_exp(x) = :(∂safe_exp($x))
eval(ForwardDiff.unary_dual_definition(:SurvivalModels, :safe_exp))