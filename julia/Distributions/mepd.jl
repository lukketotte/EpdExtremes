module MultivariateEpd

export MvEpd, repd

using Distributions, LinearAlgebra, SpecialFunctions, PDMats, Random
import Base: rand, convert, length
import Distributions: pdf, logpdf, @check_args, params, sqmahal

abstract type AbstractMvEpd <: ContinuousMultivariateDistribution end

struct GenericMvEpd{T<:Real, Cov<:AbstractPDMat, Mean<:AbstractVector} <: AbstractMvEpd
    p::T
    dim::Int
    μ::Mean
    Σ::Cov

    function GenericMvEpd{T, Cov, Mean}(p::T, dim::Int, μ::Mean, Σ::AbstractPDMat{T}) where {T, Cov, Mean}
        p > zero(p) || error("p must be positive")
        new{T, Cov, Mean}(p, dim, μ, Σ)
    end
end

function GenericMvEpd(p::T, μ::Mean, Σ::Cov) where {Cov<:AbstractPDMat, Mean<:AbstractVector,T<:Real}
    d = length(μ)
    dim(Σ) == d || throw(DimensionMismatch("The dimensions of μ and Σ are inconsistent"))
    R = Base.promote_eltype(T, μ, Σ)
    S = convert(AbstractArray{R}, Σ)
    m = convert(AbstractArray{R}, μ)
    GenericMvEpd{R, typeof(S), typeof(m)}(R(p), d, m, S)
end

function GenericMvEpd(p::Real, Σ::AbstractPDMat)
    R = Base.promote_eltype(p, Σ)
    GenericMvEpd(p, zeros(R, dim(Σ)), Σ)
end

function convert(::Type{GenericMvEpd{T}}, d::GenericMvEpd) where T <:Real
    S = convert(AbstractArray{T}, d.Σ)
    m = convert(AbstractArray{T}, d.μ)
    GenericMvEpd{T, typeof(S), typeof(m)}(T(d.p), d.dim, m, S)
end
Base.convert(::Type{GenericMvEpd{T}}, d::GenericMvEpd{T}) where {T<:Real} = d

function convert(::Type{GenericMvEpd{T}}, p, dim, μ::AbstractVector, Σ::AbstractPDMat) where T<:Real
    S = convert(AbstractArray{T}, Σ)
    m = convert(AbstractArray{T}, μ)
    GenericMvEpd{T, typeof(S), typeof(m)}(T(p), dim, m, S)
end

MvEpd(p::Real, μ::Vector{<:Real}, Σ::PDMat) = GenericMvEpd(p, μ, Σ)
MvEpd(p::Real, Σ::PDMat) = GenericMvEpd(p, Σ)
MvEpd(p::Real, μ::Vector{<:Real}, Σ::Matrix{<:Real}) = GenericMvEpd(p, μ, PDMat(Σ))
MvEpd(p::Real, Σ::Matrix{<:Real}) = GenericMvEpd(p, PDMat(Σ))

length(d::GenericMvEpd) = d.dim
params(d::GenericMvEpd) = (d.p, d.dim, d.μ, d.Σ)
sqmahal(d::GenericMvEpd, x::AbstractVector{<:Real}) = invquad(d.Σ, x - d.μ)

"""function mvepd_const(d::AbstractMvEpd)
    H = convert(eltype(d), pi^(-d.dim/2))
    H * d.dim*gamma(d.dim/2) / (gamma(1+d.dim/(2*d.p)) * 2^(1+d.dim/(2*d.p)))
end"""

function mvepd_const(d::AbstractMvEpd)
    H = convert(eltype(d), pi^(-d.dim/2))
    log(H) + log(d.dim) + loggamma(d.dim/2) - loggamma(1+d.dim/(2*d.p)) - (1+d.dim/(2*d.p))*log(2)
end

function logpdf(d::AbstractMvEpd, x::AbstractVector{T}) where T<:Real
    k = mvepd_const(d)
    mvepd_const(d) -0.5 * logdet(d.Σ) -0.5*sqmahal(d, x)^d.p
end

pdf(d::AbstractMvEpd, x::AbstractVector{<:Real}) = exp(logpdf(d, x))

function runifsphere(d::Int)
    mvnorm = rand(Normal(), d)
    mvnorm ./ sqrt.(sum(mvnorm.^2))
end

function repd(n::Int, d::GenericMvEpd)
    p, dim, μ, Σ = params(d)
    Σ = sqrt(Σ)
    res = zeros(n,dim)
    for i ∈ 1:n
        R = rand(Gamma(2,dim/(2*p))).^(1/(2*p))
        res[i,:] = μ + R*Σ*runifsphere(dim)
    end
    res
end

end
