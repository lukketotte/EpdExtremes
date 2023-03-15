using Distributed, SharedArrays
using Plots, StatsPlots
@everywhere using SpecialFunctions, LinearAlgebra, QuadGK, Roots, Distributions,Optim

@everywhere include("./Distributions/mepd.jl")
@everywhere include("./utils.jl")
@everywhere using .MultivariateEpd
@everywhere using .Utils

@everywhere f(w::Real, t::Real, β::Real, n::Int) = w^((1-n)/2-1) * (1-w)^((n-1)/2 -1) * exp(-0.5*(t/w)^β);
@everywhere g(t::Real, β::Real, n::Int) = t^((n-1)/2) * quadgk(w -> f(w, t, β, n), 0,1; atol = 2e-3)[1];
@everywhere K(β::Real, n::Int) = n*gamma(n/2)/(π^(n/2)*gamma(1+n/(2*β))*2^(1+n/(2*β)))
@everywhere df(x::Real, β::Real, n::Int) = abs(x) > 1e-10 ? g(x^2, β, n) : g(1e-20, β, n)
@everywhere dF(x::Real, β::Real, n::Int, c::Real) = quadgk(y -> c*df(y, β,n),-Inf,x; atol = 1e-4)[1]
@everywhere qF₁(x::Real, p::Real, β::Real, n::Int, c::Real) = dF(x, β, n, c) - p
@everywhere qF(p::Real, β::Real, n::Int, c::Real; intval = 20) = find_zero(x -> qF₁(x, p, β, n, c), (-intval,intval), xatol=2e-3)

@everywhere function loglik_cens(θ::AbstractVector{<:Real}, data::AbstractMatrix{<:Real}, dist::AbstractMatrix{<:Real}, thres::AbstractVector{<:Real})

  if !cond_cor([θ[1], 1., θ[2]]) # check conditions on parameters
    return 1e+10
  end

  cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), size(data, 1), size(data, 1)), [θ[1], 1.])
  if !isposdef(cor_mat)
    return 1e+10
  end
  
  #exc_ind = [i for i in 1:size(data, 1) if any(data[i, :] .> thres)]
  #ex_prob = exceedance_prob(10^6, thres, cor_mat, θ[2])
  ex_prob = 1
  exc_ind = 1:size(data,2)

  return -((1 - ex_prob) * (size(data, 2) - length(exc_ind)) + sum(logpdf(MvEpd(θ[2], cor_mat), data)))
end

@everywhere function lognorm_cens(θ::AbstractVector{<:Real}, data::AbstractMatrix{<:Real}, dist::AbstractMatrix{<:Real}, thres::AbstractVector{<:Real})

  if !cond_cor([θ[1], 1., 0.8]) # check conditions on parameters
    return 1e+10
  end

  cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), size(data, 1), size(data, 1)), [θ[1], 1.])
  if !isposdef(cor_mat)
    return 1e+10
  end
  
  #exc_ind = [i for i in 1:size(data, 1) if any(data[i, :] .> thres)]
  #ex_prob = exceedance_prob(10^6, thres, cor_mat)
  ex_prob = 1
  exc_ind = 1:size(data,2)

  return -((1 - ex_prob) * (size(data, 2) - length(exc_ind)) + sum(logpdf(MvNormal(cor_mat), data)))
end
#

@everywhere function exceedance_prob(nSims::Int, thres::AbstractVector{<:Real}, cor_mat::AbstractMatrix{<:Real}, β::Real)
  sim = repd(nSims, MvEpd(β, cor_mat))
  return length([i for i in 1:nObs if any(sim[i, :] .> thres)]) / nSims
end

@everywhere function exceedance_prob(nSims::Int, thres::AbstractVector{<:Real}, cor_mat::AbstractMatrix{<:Real})
  sim = rand(MvNormal(cor_mat), nSims)'
  return length([i for i in 1:nObs if any(sim[i, :] .> thres)]) / nSims
end

# mepd scale
dimension = 5
nObs = 250

β = 0.6
λ = 0.5
ν = 1.0
thres = 0.95
true_par = [log(λ), ν, β]
coord = rand(dimension, 2)
dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), dimension, dimension), true_par)
d = MvEpd(true_par[3], cor_mat);

ncores = 5
reps = 10
mepdRes = SharedArray{Float64}((ncores*reps,2))
normRes = SharedArray{Float64}((ncores*reps))

@sync @distributed for i in 1:(ncores*reps)
  println("iter: $i")
  #dat = repd(nObs, d)'
  dat = rand(MvTDist(3, cor_mat), nObs) 
  #dat = rand(MvNormal(cor_mat), nObs)
  thresh = quantile.(eachcol(dat), thres)
  # MEPD
  opt_res = optimize(x -> loglik_cens(x, dat, dist, thresh), [log(.5), 0.5], NelderMead())
  mepdRes[i,1] = exp(Optim.minimizer(opt_res)[1])
  mepdRes[i,2] = Optim.minimizer(opt_res)[2]

  # Normal
  opt_res = optimize(x -> lognorm_cens(x, dat, dist, thresh), [log(.5)], NelderMead())
  normRes[i] = exp(Optim.minimizer(opt_res)[1])
end

boxplot(hcat(normRes, mepdRes), labels = permutedims(["λ normal", "λ epd", "β"]))
plot!(legend=:bottomright, legendcolumns=3)

mean(normRes)
mean(mepdRes[:,1])
mean(mepdRes[:,2])