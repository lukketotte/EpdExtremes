using SpecialFunctions, LinearAlgebra, QuadGK, Roots, Distributions, Optim, Plots, Random, StatsPlots, DelimitedFiles, InvertedIndices

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

# estimate exceedance probability
function exceedance_prob(nSims::Int, thres::AbstractVector{<:Real}, cor_mat::AbstractMatrix{<:Real}, β::Real)
    sim = repd(nSims, MvEpd(β, cor_mat))
    return length([i for i in 1:nSims if any(sim[i, :] .> thres)]) / nSims
end

# censored loglikelihood
# function loglik_cens(θ::AbstractVector{<:Real}, data::AbstractMatrix{<:Real}, dist::AbstractMatrix{<:Real}, thres::AbstractVector{<:Real})
#   if !cond_cor(θ) # check conditions on parameters
#     return 1e+10
#   end
#   cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), size(data, 2), size(data, 2)), θ)
#   if !isposdef(cor_mat)
#     return 1e+10
#   end
  
#   exc_ind = [i for i in 1:size(data, 1) if any(data[i, :] .> thres)]
#   ex_prob = exceedance_prob(10^5, thres, cor_mat, θ[3])

#   return -(log((1 - ex_prob) * size(data[Not(exc_ind), :], 1)) + sum(logpdf(MvEpd(θ[3], cor_mat), data[exc_ind,:]')))
# end

# censored loglikelihood with fixed β
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

## estimate β with univariate log-likelihood
dimension = 5
nObs = 5
λ = 1.0
ν = 1.0
β = 0.4
true_par = [log(λ), ν, β]
coord = rand(dimension, 2)
dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), dimension, dimension), true_par)
d = MvEpd(β, cor_mat);
dat = repd(nObs, d)

dG1_est(param, dat) = -sum(dG1(param, dat)) # univariate log-likelihood from dG1. I changed dG1 in FFT (line 158) to return log values
β_est = optimize(x -> dG1_est(x, dat), 0.2, 0.9, show_trace = true, extended_trace = true, rel_tol = 1e-3) |> x -> Optim.minimizer(x)[1]  # estimate β with univariate log-likelihood
#


# "old" simulation without marginal transformation. not adjusted to pre-estimating β
reps = 10^3
dimension = 10
nObs = 10^4
thres = 0.95
λ = 0.5 # {0.5, 1.0}
ν = 1.0 # {1.0}
β = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# mitt (krångliga) sätt att generera unika seeds till varje replikering
num_sims = reps*length(β)
Random.seed!(trunc(Int,λ*10)+2^dimension)
seeds = rand(1:10^20, num_sims)

Threads.nthreads()
for l in eachindex(β)
  true_par = [log(λ), ν, β[l]]
  par_ests = Array{Float64}(undef, reps, 3)

  Threads.@threads for i in 1:reps
    Random.seed!(seeds[i + (l-1)*1000])
    
    coord = rand(dimension, 2)
    dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
    cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), dimension, dimension), true_par)
    dat = repd(nObs, MvEpd(true_par[3], cor_mat))
    thresh = quantile.(eachcol(dat), thres)
      
    opt_res = optimize(x -> loglik_cens(x, dat, dist, thresh), [log(1.0), 1.0, 0.5], NelderMead(), Optim.Options(g_tol = 1e-3))
    par_ests[i, 1] = exp(Optim.minimizer(opt_res)[1])
    par_ests[i, 2] = Optim.minimizer(opt_res)[2]
    par_ests[i, 3] = Optim.minimizer(opt_res)[3]
    println("Iteration: ", i, " Estimates: ", round.(par_ests[i, :], digits=4))
  end
  # skriver resultatet som en csv med namn som anger dimension och sanna parametervärden
  writedlm(join(["dim", dimension, "_", "lambda", λ, "_", "nu-", ν, "_", "beta", β[l], ".csv"], ""), par_ests, ',')
end

boxplot(par_ests[:,1], legend = false); hline!([λ], color=:red, width = 2)
boxplot(par_ests[:,2], legend = false); hline!([ν], color=:purple, width = 2)
boxplot(par_ests[:,3], legend = false); hline!([β], color=:green, width = 2)

