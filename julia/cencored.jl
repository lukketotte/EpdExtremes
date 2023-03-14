using SpecialFunctions, LinearAlgebra, QuadGK, Roots, Distributions,Optim
using Plots

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
# function loglik_cens(ν::AbstractVector{<:Real}, β::Real, data::AbstractMatrix{<:Real}, dist::AbstractMatrix{<:Real}, thres::AbstractVector{<:Real})
function loglik_cens(θ::AbstractVector{<:Real}, data::AbstractMatrix{<:Real}, dist::AbstractMatrix{<:Real}, thres::AbstractVector{<:Real})

  if !cond_cor([θ[1], 1., θ[2]]) # check conditions on parameters
    return 1e+10
  end

  cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), size(data, 2), size(data, 2)), [θ[1], 1.])
  if !isposdef(cor_mat)
    return 1e+10
  end
  
  exc_ind = [i for i in 1:size(data, 1) if any(data[i, :] .> thres)]
  ex_prob = exceedance_prob(10^6, thres, cor_mat, θ[2])

  return -((1 - ex_prob) * (size(data, 1) - length(exc_ind)) + sum(logpdf(MvEpd(θ[2], cor_mat), data')))
end
#

function exceedance_prob(nSims::Int, thres::AbstractVector{<:Real}, cor_mat::AbstractMatrix{<:Real}, β::Real)
  sim = repd(nSims, MvEpd(β, cor_mat))
  return length([i for i in 1:nObs if any(sim[i, :] .> thres)]) / nSims
end

# mepd scale
dimension = 5
nObs = 500

β = 0.8
λ = 0.5
ν = 1.0
thres = 0.95
true_par = [log(λ), β]
coord = rand(dimension, 2)
dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), dimension, dimension), true_par)
d = MvEpd(true_par[2], cor_mat);

dat = repd(nObs, d)
thresh = quantile.(eachcol(dat), thres)

loglik_cens([log(1.), 0.5], dat, dist, thresh)
opt_res = optimize(x -> loglik_cens(x, dat, dist, thresh), [log(1.), 0.5], NelderMead(), 
              Optim.Options(g_tol = 1e-3, 
                            show_trace = true, 
                            show_every = 1, 
                            extended_trace = true))
exp(Optim.minimizer(opt_res)[1])
Optim.minimizer(opt_res)[2]
#Optim.minimizer(opt_res)[3]

par_range = range(0.01, stop = 2, length = 10)
res = zeros(length(par_range))
for i in eachindex(par_range)
  res[i] = loglik_cens([log(par_range[i])], β, dat, dist, thresh)
  @show i
end
plot(par_range, res, legend=false)


# uniform scale
dimension = 5
nObs = 500

β = 0.8
thres = 0.95
true_par = [log(0.25), 1., β] # lambda, nu, p
coord = rand(dimension, 2)
dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), dimension, dimension), true_par)
d = MvEpd(true_par[3], cor_mat);

c = 2*quadgk(x -> df(x, β, dimension), 0, Inf; atol = 2e-3)[1]
data = mapslices(sortperm, repd(nObs, d); dims = 1) ./ (nObs+1)
exc_ind = [i for i in 1:nObs if any(data[i, :] .> thres)]
U = mapslices(x -> qF.(x, β, dimension, 1/c; intval = 20), data[exc_ind, :]; dims = 1)

loglik_cens([log(1)], β, c, U, dist, thres)
optimize(x -> loglik_cens(x, β, c, U, dist, thres), [log(1)], NelderMead(), Optim.Options(g_tol = 1e-8, show_trace = true, show_every = 1, extended_trace = true)) |> x -> exp.(Optim.minimizer(x))[1]

par_range = range(0.01, stop = 0.5, length = 25)
res = zeros(length(par_range))
for i in eachindex(par_range)
  res[i] = loglik_cens([log(par_range[i])], β, c, data, dist, thres)
  @show i
end
plot(par_range, res, legend=false)


