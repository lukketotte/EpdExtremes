using SpecialFunctions, LinearAlgebra, QuadGK, Roots, Distributions
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


qF₁(x::Real, prob::Real, p::Real, d::Integer) = pF(x, p, d) - prob

function upperPoint(p::Real, d::Integer)
  if p >= 0.35 && p < 0.45
    200 - 400*p + 3*d
  elseif  p >= 0.45 && p < 0.6
    46 - 68*p + 2*d
  elseif p >= 0.6 && p < 0.75
    13.3 - 14.5*p + 0.5*d
  elseif p >= 0.75
    6.3 - 5.3*p + 0.25*d
  else
    100
  end
end

qF = function(prob::Real, p::Real, d::Integer)
  prob > 0 && prob < 1 || throw(DomainError(prob, "must be on (0,1)"))
  try
    find_zero(x -> qF₁(x, prob, p, d), (0.1, upperPoint(p, d)))
  catch e
    if isa(e, DomainError) || isa(e, ArgumentError)
      find_zero(x -> qF₁(x, prob, p, d), (0.01, 100))
    end
  end
end

rF = function(n::Integer, p::Real, d::Integer)
  ret = zeros(n)
  for i in eachindex(ret)
    ret[i] = qF(rand(Uniform()), p, d)
  end
  ret
end

##