using Distributions, LinearAlgebra, SpecialFunctions, PDMats
import Base.rand
import Distributions: pdf, logpdf, @check_args, params

abstract type AbstractMvEpd <: ContinuousMultivariateDistribution end
struct GenericMvEpd{T<:Real, Cov<:AbstractPDMat{}}
