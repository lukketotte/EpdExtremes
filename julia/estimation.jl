using Distributed, SharedArrays

@everywhere using Optim, LinearAlgebra, Distributions, QuadGK, Roots
@everywhere include("./utils.jl")
@everywhere include("./FFT.jl")
@everywhere include("./Distributions/mepd.jl")
@everywhere using .MepdCopula, .Utils, .MultivariateEpd

@everywhere f(w::Real, t::Real, β::Real, n::Int) = w^((1-n)/2-1) * (1-w)^((n-1)/2 -1) * exp(-0.5*(t/w)^β);
@everywhere g(t::Real, β::Real, n::Int) = t^((n-1)/2) * quadgk(w -> f(w, t, β, n), 0,1; atol = 2e-3)[1];
@everywhere K(β::Real, n::Int) = n*gamma(n/2)/(π^(n/2)*gamma(1+n/(2*β))*2^(1+n/(2*β)))
@everywhere df(x::Real, β::Real, n::Int) = abs(x) > 1e-10 ? g(x^2, β, n) : g(1e-20, β, n)
@everywhere dF(x::Real, β::Real, n::Int, c::Real) = quadgk(y -> c*df(y, β,n),-Inf,x; atol = 1e-4)[1]

@everywhere qF₁(x::Real, p::Real, β::Real, n::Int, c::Real) = dF(x, β, n, c) - p
@everywhere qF(p::Real, β::Real, n::Int, c::Real; intval = 20) = find_zero(x -> qF₁(x, p, β, n, c), (-intval,intval), xatol=2e-3)


nObs, dimension = 50, 5;
λ, ν, β = 1.0, 1.0, 0.75
true_par = [log(λ), ν, β]

thres = 0.95
coord = rand(dimension, 2)
dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), dimension, dimension), true_par)
d = MvEpd(β, cor_mat);
dat = repd(nObs, d)
