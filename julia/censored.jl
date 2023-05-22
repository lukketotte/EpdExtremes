using SpecialFunctions, LinearAlgebra, QuadGK, Roots, Distributions, Optim, Random, Plots

include("./Distributions/mepd.jl")
include("./utils.jl")
using .MultivariateEpd
using .Utils

# OBS: β is p, following notation in MEPD paper
f(w::Real, t::Real, β::Real, n::Int) = w^((1-n)/2-1) * (1-w)^((n-1)/2 -1) * exp(-0.5*(t/w)^β);
g(t::Real, β::Real, n::Int) = t^((n-1)/2) * quadgk(w -> f(w, t, β, n), 0,1; atol = 2e-3)[1];
K(β::Real, n::Int) = n*gamma(n/2)/(π^(n/2)*gamma(1+n/(2*β))*2^(1+n/(2*β)))
df(x::Real, β::Real, n::Int) = abs(x) > 1e-10 ? g(x^2, β, n) : g(1e-20, β, n)
dF(x::Real, β::Real, n::Int, c::Real) = quadgk(y -> c*df(y, β,n),-Inf,x; atol = 1e-4)[1]

qF₁(x::Real, p::Real, β::Real, n::Int, c::Real) = dF(x, β, n, c) - p
qF(p::Real, β::Real, n::Int, c::Real; intval = 20) = find_zero(x -> qF₁(x, p, β, n, c), (-intval,intval), xatol=2e-3)
#

# censored loglikelihood
function loglik_cens(θ::AbstractVector{<:Real}, data::AbstractMatrix{<:Real}, dist::AbstractMatrix{<:Real}, thres::AbstractVector{<:Real})

  if !cond_cor(θ) # check conditions on parameters
    return 1e+10
  end

  cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), size(data, 2), size(data, 2)), θ)
  if !isposdef(cor_mat)
    return 1e+10
  end
  
  exc_ind = [i for i in 1:size(data, 1) if any(data[i, :] .> thres)]
  ex_prob = exceedance_prob(10^5, thres, cor_mat, θ[3])

  return -(log((1 - ex_prob) * (size(data, 1) - length(exc_ind))) + sum(logpdf(MvEpd(θ[3], cor_mat), data[exc_ind, :]')))
end
#

function exceedance_prob(nSims::Int, thres::AbstractVector{<:Real}, cor_mat::AbstractMatrix{<:Real}, β::Real)
  sim = repd(nSims, MvEpd(β, cor_mat))
  return length([i for i in 1:nSims if any(sim[i, :] .> thres)]) / nSims
end

# mepd scale
dimension = 5
nObs = 1000

λ = 0.5
ν = 1.0
β = 0.9
true_par = [log(λ), ν, β]

Random.seed!(123)
coord = rand(dimension, 2)
dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), dimension, dimension), true_par)
d = MvEpd(β, cor_mat);
dat = repd(nObs, d)

thres = 0.0
thresh = quantile.(eachcol(dat), thres)

loglik_cens(true_par, dat, dist, thresh)
opt_res = optimize(x -> loglik_cens(x, dat, dist, thresh), [log(1.0), 0.5, 0.5], NelderMead(), 
              Optim.Options(g_tol = 1e-2, 
                            show_trace = true, 
                            show_every = 5, 
                            extended_trace = true))
exp(Optim.minimizer(opt_res)[1])
Optim.minimizer(opt_res)[2]
Optim.minimizer(opt_res)[3]



############## uniform scale
function loglik_cens(θ::AbstractVector{<:Real}, β::Real, data::AbstractMatrix{<:Real}, dist::AbstractMatrix{<:Real}, thres::AbstractVector{<:Real})
  if !cond_cor(θ) # check conditions on parameters
    return 1e+10
  end
  
  cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), size(data, 2), size(data, 2)), θ)
  if !isposdef(cor_mat)
    return 1e+10
  end
  
  exc_ind = [i for i in 1:size(data, 1) if any(data[i, :] .> thres)]
  ex_prob = exceedance_prob(10^5, thres, cor_mat, β)

  return -(log((1 - ex_prob) * (size(data, 1) - length(exc_ind))) + sum(logpdf(MvEpd(β, cor_mat), data[exc_ind,:]')))
end
#

function exceedance_prob(nSims::Int, thres::AbstractVector{<:Real}, cor_mat::AbstractMatrix{<:Real}, β::Real)
  sim = repd(nSims, MvEpd(β, cor_mat))
  return length([i for i in 1:nSims if any(sim[i, :] .> thres)]) / nSims
end

dimension = 5
nObs = 100

β = 0.6
thres = 0.95
true_par = [log(1.0), 1.0] # lambda, nu, p
coord = rand(dimension, 2)
dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), dimension, dimension), true_par)
d = MvEpd(β, cor_mat);

c = 2*quadgk(x -> df(x, β, dimension), 0, Inf; atol = 2e-3)[1]
data_U = mapslices(r -> invperm(sortperm(r, rev=false)), repd(nObs, d); dims = 1) ./ (nObs+1)
thres_U = quantile.(eachcol(data_U), thres)
data = mapslices(x -> qF.(x, β, dimension, 1/c; intval = 20), data_U; dims = 1)
thresh = repeat([qF(thres_U[1], β, dimension, 1/c; intval = 20)], dimension)

loglik_cens(true_par, β, data, dist, thresh)
res = optimize(x -> loglik_cens(x, β, data, dist, thresh), [log(2.0), 0.5], NelderMead(), 
                          Optim.Options(g_tol = 1e-3, 
                          show_trace = true, 
                          show_every = 5, 
                          extended_trace = true))
[exp(Optim.minimizer(res)[1]), Optim.minimizer(res)[2]]




