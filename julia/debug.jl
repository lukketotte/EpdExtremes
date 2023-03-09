using SpecialFunctions, LinearAlgebra, QuadGK, Roots, Distributions
using Plots

include("./Distributions/mepd.jl")
using .MultivariateEpd

# OBS: β is p, following notation in MEPD paper
f(w::Real, t::Real, β::Real, n::Int) = w^((1-n)/2-1) * (1-w)^((n-1)/2 -1) * exp(-0.5*(t/w)^β);
g(t::Real, β::Real, n::Int) = t^((n-1)/2) * quadgk(w -> f(w, t, β, n), 0,1; atol = 1e-5)[1];
K(β::Real, n::Int) = n*gamma(n/2)/(π^(n/2)*gamma(1+n/(2*β))*2^(1+n/(2*β)))
df(x::Real, β::Real, n::Int) = abs(x) > 1e-10 ? g(x^2, β, n) : g(1e-20, β, n)
dF(x::Real, β::Real, n::Int, c::Real) = quadgk(y -> c*df(y, β,n),-Inf,x; atol = 1e-5)[1]

qF₁(x::Real, p::Real, β::Real, n::Int, c::Real) = dF(x, β, n, c) - p
qF(p::Real, β::Real, n::Int, c::Real; intval = 20) = find_zero(x -> qF₁(x, p, β, n, c), (-intval,intval), xatol=2e-3)


## Estimation
function loglik(par::AbstractVector{<:Real}, β::Real, c::Real, data::AbstractMatrix{<:Real}, dist::AbstractMatrix{<:Real})
  if !cond_cor([par..., β]) # check conditions on parameters
    return 1e+10
  end
  cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), size(data, 2), size(data, 2)), par)
  -sum(logpdf(MvEpd(β, cor_mat), data'))+sum(log.(df.(reshape(data, (prod(size(data)),)), β, 2) ./c))
end

function loglik(ν::AbstractVector{<:Real}, β::Real, c::Real, data::AbstractMatrix{<:Real}, dist::AbstractMatrix{<:Real})
  if !cond_cor([ν[1],1.,β]) # check conditions on parameters
    return 1e+10
  end
  cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), size(data, 2), size(data, 2)), [ν[1],1])
  -sum(logpdf(MvEpd(β, cor_mat), data'))+sum(log.(df.(reshape(data, (prod(size(data)),)), β, size(data, 2)) ./c))
end

# normal case
function loglikNormal(ν::AbstractVector{<:Real}, data::AbstractMatrix{<:Real}, dist::AbstractMatrix{<:Real})
  U = mapslices(x -> quantile.(Normal(), x), dat; dims = 1)
  cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), size(data, 2), size(data, 2)), [ν[1],1])
  -sum(logpdf(MvNormal(cor_mat), U')) + sum(logpdf.(Normal(), reshape(U, (prod(size(data)),))))
end

β = 0.8
c = 2*quadgk(x -> df(x, β, dimension), 0, Inf; atol = 2e-3)[1]

dat = mapslices(sortperm, repd(nObs, d); dims = 1) ./ (nObs+1)
U = mapslices(x -> qF.(x, β, dimension, 1/c; intval = 20), dat; dims = 1);

optimize(x -> loglikNormal(x, dat, dist), [log(4)], NelderMead()) |> x -> Optim.minimizer(x)
optimize(x -> loglik(x, β, c, U, dist), [log(4)], NelderMead()) |> x -> Optim.minimizer(x)

##


# test
d = MvEpd(0.7, [1. 0.2 ; 0.2 1]);
n = 200
X = mapslices(sortperm, repd(n,d); dims = 1) ./ (n+1)
scatter(X[:, 1], X[:, 2])

vals = zeros(6)
# check vals
β = 0.5
c = 2*quadgk(x -> df(x, β, 2), 0, Inf; atol = 2e-3)[1]
dF(30, β, 2, 1/c) # make sure its close to 1
U = mapslices(x -> qF.(x, β, 2, 1/c; intval = 40), X; dims = 1);
vals[1] = sum(logpdf(d, U'))-sum(log.(df.(reshape(U, (prod(size(U)),)), β, 2) ./c))

β = 0.65
c = 2*quadgk(x -> df(x, β, 2), 0, Inf; atol = 2e-3)[1]
#dF(30, 0.65, 2, 1/c) # make sure its close to 1
U = mapslices(x -> qF.(x, β, 2, 1/c; intval = 30), X; dims = 1);
vals[2] = sum(logpdf(d, U'))-sum(log.(df.(reshape(U, (prod(size(U)),)), β, 2) ./c))

β = 0.7
c = 2*quadgk(x -> df(x, β, 2), 0, Inf; atol = 2e-3)[1]
U = mapslices(x -> qF.(x, β, 2, 1/c), X; dims = 1);
vals[3] = sum(logpdf(d, U'))-sum(log.(df.(reshape(U, (prod(size(U)),)), β, 2) ./c))

β = 0.8
c = 2*quadgk(x -> df(x, β, 2), 0, Inf; atol = 2e-3)[1]
#dF(18, 0.8, 2, 1/c)
U = mapslices(x -> qF.(x, β, 2, 1/c; intval = 18), X; dims = 1);
vals[4] = sum(logpdf(d, U'))-sum(log.(df.(reshape(U, (prod(size(U)),)), β, 2) ./c))

β = 0.9
c = 2*quadgk(x -> df(x, β, 2), 0, Inf; atol = 2e-3)[1]
#dF(5, 0.9, 2, 1/c)
U = mapslices(x -> qF.(x, β, 2, 1/c; intval = 5), X; dims = 1);
vals[5] = sum(logpdf(d, U'))-sum(log.(df.(reshape(U, (prod(size(U)),)), β, 2) ./c))

β = 0.95
c = 2*quadgk(x -> df(x, β, 2), 0, Inf; atol = 2e-3)[1]
#dF(5, β, 2, 1/c)
U = mapslices(x -> qF.(x, β, 2, 1/c; intval = 5), X; dims = 1);
vals[6] = sum(logpdf(d, U'))-sum(log.(df.(reshape(U, (prod(size(U)),)), β, 2) ./c))

plot(vals)
