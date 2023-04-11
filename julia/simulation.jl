using SpecialFunctions, LinearAlgebra, QuadGK, Roots, Distributions, Optim, Plots, Random, StatsPlots, Serialization

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
    return length([i for i in 1:nObs if any(sim[i, :] .> thres)]) / nSims
end

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
  ex_prob = exceedance_prob(10^6, thres, cor_mat, θ[3])

  return -((1 - ex_prob) * (size(data, 1) - length(exc_ind)) + sum(logpdf(MvEpd(θ[3], cor_mat), data')))
end
#

# simulation
reps = 1000
dimension = 10
nObs = 1000

thres = 0.95
λ = 1.0 # {0.5, 1}
ν = 1.0 # {1.0}
β = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# true_par = [log(λ), ν, β]

for l in eachindex(β)
  true_par = [log(λ), ν, β[l]]

  dist_mat = Vector{Matrix{Float64}}(undef, reps)
  dat_mat = Vector{Matrix{Float64}}(undef, reps)
  thres_mat = Vector{Vector{Float64}}(undef, reps)
  Random.seed!(123)
  for j in 1:reps
    coord = rand(dimension, 2)
    dist_mat[j] = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
    cor_mat = cor_fun(reshape(sqrt.(dist_mat[j][1, :] .^ 2 .+ dist_mat[j][2, :] .^ 2), dimension, dimension), true_par)
    d = MvEpd(true_par[3], cor_mat)
    dat_mat[j] = repd(nObs, d)
    thres_mat[j] = quantile.(eachcol(dat_mat[j]), thres)
  end

  par_ests = Array{Float64}(undef, reps, 3)
  Random.seed!(789)
  Threads.@threads for i in 1:reps
    opt_res = optimize(x -> loglik_cens(x, dat_mat[i], dist_mat[i], thres_mat[i]), [log(1.0), 1.0, 0.5], NelderMead(), Optim.Options(g_tol = 1e-2))
    par_ests[i, 1] = exp(Optim.minimizer(opt_res)[1])
    par_ests[i, 2] = Optim.minimizer(opt_res)[2]
    par_ests[i, 3] = Optim.minimizer(opt_res)[3]

    println("Iteration: ", i, " Estimates: ", round.(par_ests[i, :], digits=4))
  end
  serialize(join(["dim-", dimension, "_", "lambda-", λ, "_", "nu-", ν, "_", "beta-", β[l], ".dat"], ""), par_ests)
  @show l
end

boxplot(par_ests[:,1], legend = false); hline!([λ], color=:red, width = 2)
boxplot(par_ests[:,2], legend = false); hline!([ν], color=:purple, width = 2)
boxplot(par_ests[:,3], legend = false); hline!([β], color=:green, width = 2)

