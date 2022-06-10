using LinearAlgebra, SpecialFunctions, QuadGK, Distributions, Optim, Roots
using Plots

h05 = function(r::Real, n::Integer)
    C = 2^((3*n + 1)/2)*gamma((n+1)/2)
    r^n * exp(-r^2/8) / C
end

f(r::Real, x::Real, n::Integer) = pdf(Normal(0, r), x) * h05(r, n)
f(x::Real, n::Integer) = quadgk(r -> f(r, x, n), 0, Inf)[1]

F(r::Real, x::Real, n::Integer) = cdf(Normal(0, 1), x/r) * h05(r, n)
F(x::Real, n::Integer) = quadgk(r -> F(r, x, n), 0, Inf)[1]

qroot(x::Real, q::Real, n::Int) = F(x, n) - q
Q(q::Real, n::Int) = find_zero((x -> qroot(x, q, n)), quantile(Laplace(), q), Order16())
