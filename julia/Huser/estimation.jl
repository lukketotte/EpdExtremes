using Distributed, SharedArrays

@everywhere using Optim, Compat, LinearAlgebra, Statistics, Random, Dates
@everywhere include("../utils.jl")
@everywhere include("./FFThuser.jl")
@everywhere using .HuserCopula, .Utils

dimension = 2
nObs = 100

Random.seed!(321)
true_par = [log(2.0), 1, 1., 1.] # lambda, nu, p
coord = rand(dimension, 2)
dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), dimension, dimension), true_par)
dat = rC(nObs, cor_mat, true_par)
(n, D) = size(dat)

@time x = optimize(x -> nllik(x, dat, coord, n, D, 5),true_par, NelderMead(), 
                   Optim.Options(g_tol = 2e-3, # default 1e-8
                                 show_trace = true,
                                 show_every = 1,
                                 extended_trace = true)
                    )

x.minimizer
x.minimum



function nllik(param::Vector{Float64}, dat::Matrix{Float64}, coord::Matrix{Float64}, n::Integer, D::Integer, ncores::Integer)
  if !cond_cor(param) # check conditions on parameters
      return 1e+10
  end

  # compute the matrix of correlations in W
  dists = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
  Sigmab = cor_fun(reshape(sqrt.(dists[1, :] .^ 2 .+ dists[2, :] .^ 2), D, D), param)
  if !isposdef(Sigmab)
      return 1e+10
  end

  nllik_res = SharedArray{Float64}(ncores)
  @sync @distributed for i in 1:ncores # ncores can be no larger than the number of observations
      nllik_res[i] = nllik_block(i, dat, param, Sigmab, n, ncores)
  end
  if any(isnan.(nllik_res))
      return 1e+10
  else
      return sum(nllik_res)
  end
end

@everywhere function nllik_block(block::Integer, dat::Matrix{Float64}, param::Vector{Float64}, Sigmab::Matrix{Float64}, n::Integer, ncores::Integer)
  if ncores > 1
      indmin = vcat(0.5, quantile(1:n, LinRange(1 / ncores, (ncores - 1) / ncores, ncores - 1)))[block]
      indmax = vcat(quantile(1:n, LinRange(1 / ncores, (ncores - 1) / ncores, ncores - 1)), n + 0.5)[block]
      ind_block = round.(Int, LinRange(1, n, n)[(1:n .> indmin) .&& (1:n .≤ indmax)]) # indices of the specific block
  elseif ncores == 1
      ind_block = 1:n
  end
  contrib = dC(reshape(dat[ind_block, :], length(ind_block), size(Sigmab, 1)), Sigmab, param[3:4])
  return -sum(contrib)
end

##
using Plots

β = range(0.6, 1.4, length = 20)
res = zeros(Float64, 20)
for i in eachindex(res)
  res[i] = nllik([1.41, 0.44, β[i], 0.3], dat, coord, n, D, 5)
end


plot(β, res)