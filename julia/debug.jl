using Distributions, QuadGK, LinearAlgebra, Roots, SpecialFunctions

g(x::Float64, p::Real) = exp(-abs(x)^p/2) / (π * gamma(1+1/p) * 2^(1/p))
f(y::Float64, x::Float64, p::Real) = (y-x^2 + 1.e-20)^(-1/2) * g(y, p)
depd(x::Float64, p::Real) = quadgk(y -> f(y, x, p), x^2, Inf; atol=3e-2)[1]
pepd(x::Float64, p::Real) = 1/2 + quadgk(y -> depd(y, p), 0, x; atol=3e-2)[1]

# dG1 and depd should give the same_vec
dG1([2. 12.], 0.9)
depd(12., 0.9)



function qepd(prob::Real, p::Real)
  f(x) = pepd(x, p) - prob
  #find_zero(f, x0, Order16())
  try
    find_zero(x -> f(x), [-15, 15], xatol=2e-3)
  catch e
    find_zero(x -> f(x), [-500, 500], xatol=2e-3)
  end
end

function C1(x::Float64, y::Float64, p::Real, ρ::Real)
  1/√(1-ρ^2) * g((x^2 + y^2 - 2*ρ*x*y)/(1-ρ^2), p)
end
inner(v::Real, y::Real, p::Real, ρ::Real) = quadgk(x -> C1(y, x, p, ρ), -Inf, qepd(v, p); atol=3e-2)[1]
C(u::Real, v::Real, p::Real, ρ::Real) = quadgk(y -> inner(v, y, p, ρ), -Inf, qepd(u, p); atol=3e-2)[1]
@time C(0.9, 0.9, 0.9, 0.5)


C1(x::Float64, p::Real, ρ::Real, q::Real) = quadgk(y -> C(y, x, p, ρ), -Inf, q)[1]
C2(p::Real, ρ::Real, q::Real) = quadgk(y -> C1(y, p, ρ, q), -Inf, q; rtol=1e-12)[1]