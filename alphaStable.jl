module AlphaStableDistribution

export AlphaStable

using Distributions, LinearAlgebra, SpecialFunctions, Random, QuadGK
import Base.rand
import Distributions: pdf, logpdf, @check_args, params

struct AlphaStable{T} <: Distributions.ContinuousUnivariateDistribution
    α::T
    β::T
    σ::T
    μ::T
    AlphaStable{T}(α::T, β::T, σ::T, μ::T) where {T} = new{T}(α, β, σ, μ)
end

function AlphaStable(α::T, β::T, σ::T, μ::T; check_args::Bool=true) where {T <: Real}
    @check_args(AlphaStable, zero(α) < α < one(α))
    @check_args(AlphaStable, -one(β) < β <= one(β))
    @check_args(AlphaStable, zero(σ) < σ)
    return AlphaStable{T}(α, β, σ, μ)
end

function AlphaStable(α::T; check_args::Bool=true) where {T <: Real}
    γ = 2^(1-1/α)*cos(π*α/2)
    return AlphaStable(promote(α, 1., γ, γ*tan(π*α/2))...)
end

function AlphaStable(α::T, σ::T; check_args::Bool=true) where {T <: Real}
    γ = (σ*cos(π*α/2))
    return AlphaStable(promote(α, 1., γ, γ*tan(π*α/2))...)
end

params(d::AlphaStable) = (d.α, d.β, d.σ, d.μ)

function chf(t, d::AlphaStable{T}) where {T<:AbstractFloat}
    α, β, σ, μ = params(d)
    Φ = α == one(T) ? -2*log(abs(t)) : tan(π*α/2)
    exp(t*im*μ - abs(σ*t)^α*(1-im*β*sign(t) * Φ))
end

#pdf(d::AlphaStable, x::Real) = real(quadgk(t -> exp(-im*t*x)*chf(t, d), -Inf, Inf)[1])
# following Nolan 1997
h(t::Real, x::Real, α::Real, β::Real) = x*t -β*tan(π*α/2) * t^α
function pdf(d::AlphaStable, x::Real)
    α, β, σ, μ = params(d)
    x = (x-μ)/σ
    (quadgk(t -> exp(-t^α)*cos(h(t, x + β*tan(π*α/2), α, β)), 0, Inf)[1])π
end

logpdf(d::AlphaStable, x::Real) = log(pdf(d, x))

function rand(rng::AbstractRNG, d::AlphaStable{T}) where {T<:AbstractFloat}
    α, β, σ, μ = params(d)
    ϕ = (rand(rng, T) - 0.5) * π
    if α == one(T) && β == zero(T)
        return μ + σ * tan(ϕ)
    end
    w = -log(rand(rng, T))
    β == zero(T) && (return μ + σ * ((cos((1-α)*ϕ) / w)^(one(T)/α - one(T)) * sin(α * ϕ) / cos(ϕ)^(one(T)/α)))
    cosϕ = cos(ϕ)
    if abs(α - one(T)) > 1e-8
        ζ = β * tan(π * α / 2)
        aϕ = α * ϕ
        a1ϕ = (one(T) - α) * ϕ
        return μ + σ * (( (sin(aϕ) + ζ * cos(aϕ))/cosϕ * ((cos(a1ϕ) + ζ*sin(a1ϕ))) / ((w*cosϕ)^((1-α)/α)) ))
    end
    bϕ = π/2 + β*ϕ
    x = 2/π * (bϕ * tan(ϕ) - β * log(π/2*w*cosϕ/bϕ))
    α == one(T) || (x += β * tan(π*α/2))

    return μ + σ * x
end

end
