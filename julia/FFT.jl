using SpecialFunctions, LinearAlgebra,QuadGK,Roots
using BenchmarkTools

ζ(α::Real) = -tan(π*α/2)
θ₀(α::Real) = 1/α * atan(tan(π*α/2))
function V(α::Real, θ::Real)
  ϑ = θ₀(α)
  a = cos(α*ϑ)^(1/(α-1))
  b = (cos(θ)/sin(α*(ϑ + θ)))^(α/(α-1))
  c = cos(α*ϑ + (α-1)*θ) / cos(θ)
  return a*b*c 
end

h(θ::Real, x::Real, α::Real) = (x-ζ(α))^(α/(α-1))*V(α,θ)*exp(-(x-ζ(α))^(α/(α-1))*V(α,θ))
f(x::Real, α::Real) = α/(π*(x-ζ(α))*abs(α-1)) * quadgk(θ -> h(θ, x, α), -θ₀(α), π/2)[1]
dstable(x::Real, α::Real, γ::Real) = f((x-γ * tan(π*α/2))/γ, α)/γ

dF = function(x::Real, p::Real, d::Int)
  p > 0 && p < 1 || throw(DomainError(p,"must be on (0,1)")) 
  γ = 2^(1-1/p) * cos(π*p/2)^(1/p)
  C = 2^(1+d/2*(1-1/p)) * gamma(1+d/2) / gamma(1+d/(2*p))
  x > 0 ? C*x^(d-3)*dstable(x^(-2), p, γ) : 0
end

pF = function(x::Real, p::Real, d::Int)
  quadgk(x -> dF(x,p,d), 0, x)[1]
end

@time pF(2, 0.5, 1)

test(x::Real) = pF(x, 0.5, 1) - 0.8

pF(1, 0.5, 1) - 0.8


find_zero(test, (1, 10))
