# SurvivalModels

[![Build Status](https://github.com/aidanlsb/SurvivalModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/aidanlsb/SurvivalModels.jl/actions/workflows/CI.yml?query=branch%3Amain)

Quick start:
```
using SurvivalModels

mc = MixtureCureEstimator(WeibullEstimator())

res = fit(mc, t, e, X)
confint(res)
```

Where `t` is a vector of durations, `e` is a vector of booleans indicating if the event was observed or not, and `X` is a matrix of covariates.
