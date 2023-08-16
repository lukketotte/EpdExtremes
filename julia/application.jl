using Distributed
# addprocs(6)
using SharedArrays, CSV, Random, DelimitedFiles, Tables

@everywhere using Optim, LinearAlgebra, Distributions, QuadGK, Roots
@everywhere include("./utils.jl")
@everywhere include("./FFT.jl")
@everywhere include("./Distributions/mepd.jl")
@everywhere include("./Huser/FFThuser.jl")
@everywhere using .MepdCopula, .Utils, .MultivariateEpd, .HuserCopula

@everywhere f(w::Real, t::Real, β::Real, n::Int) = w^((1-n)/2-1) * (1-w)^((n-1)/2 -1) * exp(-0.5*(t/w)^β);
@everywhere g(t::Real, β::Real, n::Int) = t^((n-1)/2) * quadgk(w -> f(w, t, β, n), 0,1; atol = 2e-3)[1];
@everywhere K(β::Real, n::Int) = n*gamma(n/2)/(π^(n/2)*gamma(1+n/(2*β))*2^(1+n/(2*β)))
@everywhere df(x::Real, β::Real, n::Int) = abs(x) > 1e-10 ? g(x^2, β, n) : g(1e-20, β, n)
@everywhere dF(x::Real, β::Real, n::Int, c::Real) = quadgk(y -> c*df(y, β,n),-Inf,x; atol = 1e-4)[1]

@everywhere qF₁(x::Real, p::Real, β::Real, n::Int, c::Real) = dF(x, β, n, c) - p
@everywhere qF(p::Real, β::Real, n::Int, c::Real; intval = 20) = find_zero(x -> qF₁(x, p, β, n, c), (-intval,intval), xatol=2e-3)

@everywhere function dfmarg(β::AbstractVector{<:Real}, data::AbstractMatrix{<:Real})
    if β[1] < 0.95 && β[1] > 0.3
      c = 2*quadgk(x -> df(x, β[1], size(data,2)), 0, Inf; atol = 1e-3)[1]
      return -sum(log.((df.(vec(data), β[1], size(data, 2))) ./ c))
    else
      return 1e+10
    end
end

@everywhere function marg_fun(β) return dfmarg([β], data) end # enabling univariate optimisation of β

@everywhere function exceedance_prob(nSims::Int, thres::Real, cor_mat::AbstractMatrix{<:Real}, β::Real)
  exceedance = 0
  for j in 1:trunc(Int, nSims/10000)
    sim = repd(10000, MvEpd(β, cor_mat))
    exceedance += length([i for i in 1:size(sim, 1) if any(sim[i, :] .> thres)])
  end
  return exceedance / nSims

  ## faster but allocates a lot of memory
  # sim = repd(nSims, MvEpd(β, cor_mat))
  # return length([i for i in 1:nSims if any(sim[i, :] .> thres)]) / nSims
end

@everywhere function loglik_cens(θ::AbstractVector{<:Real}, β::Real, data::AbstractMatrix{<:Real}, 
    data_exc::AbstractMatrix{<:Real}, dist::AbstractMatrix{<:Real}, thres::Real; isotropic::Bool)

    if !cond_cor(θ) # check conditions on parameters
      return 1e+10
    end

    if isotropic
      dists = reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), size(data, 2), size(data, 2))
    else
      angle_mat = reshape([cos(θ[3]), sin(θ[3]), -sin(θ[3]), cos(θ[3])], 2, 2)
      Ω = angle_mat * reshape([1, 0, 0, θ[4]^(-2)], 2, 2) * transpose(angle_mat)
      dists_mahal = zeros(size(dist, 2))
      for i in eachindex(dist[1,:])
          dists_mahal[i] = transpose(dist[:,i]) * inv(Ω) * dist[:,i]
      end
      dists = reshape(dists_mahal, size(data, 2), size(data, 2))
    end
    
    cor_mat = cor_fun(dists, θ)
    if !isposdef(cor_mat)
      return 1e+10
    end

    ex_prob = exceedance_prob(trunc(Int, 1e6), thres, cor_mat, β)
    
    return -(log(1 - ex_prob) * (size(data, 1) - size(data_exc, 1)) + sum(logpdf(MvEpd(β, cor_mat), permutedims(data_exc))))
end




####################
####################
####################

# pwd()
cd("C:\\Users\\aleen962\\Dropbox\\PhD\\Forskning\\Power exponential dist extremes\\application\\data\\data_sets")
coord = convert(Matrix{Float64}, readdlm("wind_gust_coordinates_km.csv", ',')[2:end,:]) # lon, lat
dimension = size(coord, 1)
dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]));
data = convert(Matrix{Float64}, readdlm("model_data_complete.csv", ',')[2:end,:])

### threshold and exceedances
thres_q = 0.98;
thres = quantile(vec(data), thres_q)
exc_ind = [i for i in 1:size(data, 1) if any(data[i, :] .> thres)]

###############
##### EPD #####
###############
# @time opt_res = optimize(x -> dfmarg([x], data), 0.3, 0.95, rel_tol = 1e-3, show_trace = true, show_every = 1) # CHANGED: univariate optimisation of β bcs there was some issue with the multivariate optimisation for β=0.4
# βhat = Optim.minimizer(opt_res)[1]
# writedlm("betaHat.csv", βhat, ',') # save βhat as .csv

### rank transform data to approximately u(0,1)
data_U = mapslices(r -> invperm(sortperm(r, rev=false)), data; dims = 1) ./ (size(data, 1) + 1) # data transformed to (pseudo)uniform(0,1)
# c = 2*quadgk(x -> df(x, βhat, dimension), 0, Inf; atol = 2e-3)[1] # constant

### transform data from u(0,1) to mepd scale - took 14.7 hours for one bootstrap sample with threshold 0.98
# data_exc = zeros(size(vec(data_U[exc_ind,:]), 1))
# using Dates; Dates.now()
# @time Threads.@threads for i in 1:size(vec(data_U[exc_ind,:]), 1)
#   println(i)
#   data_exc[i] = qF(vec(data_U[exc_ind,:])[i], βhat, dimension, 1/c; intval = 40) 
# end
# data_exc = reshape(data_exc, size(data_U[exc_ind,:]))

# save transformed data
# using Tables
# cd("C:\\Users\\aleen962\\Dropbox\\PhD\\Forskning\\Power exponential dist extremes\\application")
# CSV.write("gustData_mepdTrans_u098.csv", Tables.table(data_exc))
#

### fit mepd likelihood
# load transformed data and βhat
cd("C:\\Users\\aleen962\\Dropbox\\PhD\\Forskning\\Power exponential dist extremes\\application")
data_exc = convert(Matrix{Float64}, readdlm("gustData_mepdTrans_u098.csv", ',')[2:end,:])
βhat = readdlm("betaHat.csv", ',')[1]

### isotropic correlation model
using Dates; Dates.now()
opt_res = optimize(x -> loglik_cens(x, βhat, data, data_exc, dist, thres; isotropic=true), [log(100.), 1.], NelderMead(), 
    Optim.Options(g_tol = 1e-2, iterations = 100, show_trace = true, show_every = 10, extended_trace = true))
#
Optim.minimizer(opt_res)[1], Optim.minimizer(opt_res)[2]

# log-likelihood value
logL_iso = -(loglik_cens(Optim.minimizer(opt_res), βhat, data, data_exc, dist, thres; isotropic=true) -dfmarg([βhat], data_exc))
# AIC
aic_iso = 2*((length(Optim.minimizer(opt_res))+1) + (loglik_cens(Optim.minimizer(opt_res), βhat, data, data_exc, dist, thres; isotropic=true) -dfmarg([βhat], data_exc)))
#
appl_results = [Optim.minimizer(opt_res)..., βhat, logL_iso, aic_iso]
# writedlm("application_results_isotropic.csv", appl_results, ',') # save results as .csv


### anisotropic correlation model
using Dates; Dates.now()
opt_res = optimize(x -> loglik_cens(x, βhat, data, data_exc, dist, thres; isotropic=false), [log(100.), 1., 1., 1.], NelderMead(), 
    Optim.Options(g_tol = 1e-2, iterations = 300, show_trace = true, show_every = 10, extended_trace = true))
#
Optim.minimizer(opt_res)[1], Optim.minimizer(opt_res)[2], Optim.minimizer(opt_res)[3], Optim.minimizer(opt_res)[4]

# log-likelihood value
logL_aniso = -(loglik_cens(Optim.minimizer(opt_res), βhat, data, data_exc, dist, thres; isotropic=false) -dfmarg([βhat], data_exc))
# AIC
aic_aniso = 2*((length(Optim.minimizer(opt_res))+1) + (loglik_cens(Optim.minimizer(opt_res), βhat, data, data_exc, dist, thres; isotropic=false) -dfmarg([βhat], data_exc)))
#
appl_results = [Optim.minimizer(opt_res)..., βhat, logL_aniso, aic_aniso]
# writedlm("application_results_anisotropic.csv", appl_results, ',') # save results as .csv



#############################
### calculate standard errors

## sandwich estimator
@everywhere function loglik_cens_gr(θ::AbstractVector{<:Real}, β::Real, data::AbstractVector{<:Real}, obs::Real, exc_ind::AbstractVector{<:Real}, dist::AbstractMatrix{<:Real}, thres::Real; isotropic::Bool)

  if !cond_cor(θ) # check conditions on parameters
    return 1e+10
  end

  if isotropic
      dists = reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), length(data), length(data))
      # dists = reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), size(data, 2), size(data, 2))
  else
      angle_mat = reshape([cos(θ[3]), sin(θ[3]), -sin(θ[3]), cos(θ[3])], 2, 2)
      Ω = angle_mat * reshape([1, 0, 0, θ[4]^(-2)], 2, 2) * transpose(angle_mat)
      dists_mahal = zeros(size(dist, 2))
      for i in 1:size(dist, 2)
          dists_mahal[i] = transpose(dist[:,i]) * inv(Ω) * dist[:,i]
      end
      dists = reshape(dists_mahal, length(data), length(data))
  end

  cor_mat = cor_fun(dists, θ)
  if !isposdef(cor_mat)
    return 1e+10
  end

  if obs ∈ exc_ind
      return logpdf(MvEpd(β, cor_mat), data)
  else
      ex_prob = exceedance_prob(trunc(Int, 1e7), thres, cor_mat, β)
      return log(1 - ex_prob)
  end
end

data_sd = data
data_sd[exc_ind,:] = data_exc

cd("C:\\Users\\aleen962\\Dropbox\\PhD\\Forskning\\Power exponential dist extremes\\application")
pars = readdlm("application_results_isotropic.csv", ',')[1:2] # isotropic
# pars = readdlm("application_results_anisotropic.csv", ',')[1:4] # anisotropic

# sandwich estimator
reps = size(data_sd, 1)
score = zeros(reps, length(pars))
Threads.@threads for k in 1:reps
  println(k)
  score[k,:] = FiniteDiff.finite_difference_jacobian(x -> loglik_cens_gr(x, βhat, data2[k,:], k, exc_ind, dist, thres; isotropic = false), pars)
end
score




########################################################################################
### moving block bootstrap to calculate confidence intervals for the parameter estimates

# load data
cd("C:\\Users\\aleen962\\Dropbox\\PhD\\Forskning\\Power exponential dist extremes\\application\\data\\data_sets")
coord = convert(Matrix{Float64}, readdlm("wind_gust_coordinates_km.csv", ',')[2:end,:]) # lon, lat
dimension = size(coord, 1)
dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]));
data = convert(Matrix{Float64}, readdlm("model_data_complete.csv", ',')[2:end,:])
thres_q = 0.98

using Dates, Printf

iso = false
reps = 6
if iso == false
  boot_par_ests = SharedArray{Float64}(reps, 5)
else
  boot_par_ests = SharedArray{Float64}(reps, 3)
end
Random.seed!(789)
@time @sync @distributed for j in 1:reps

 println("iteration: ", j, " start, ", "time: ", Dates.now())

  boot_sample = block_boot_sample(data, (24*7))

  # threshold and exceedance indices
  thres = quantile(vec(boot_sample), thres_q)
  exc_ind = [i for i in 1:size(boot_sample, 1) if any(boot_sample[i, :] .> thres)]

  # estimate β
  opt_res_beta = optimize(x -> dfmarg([x], boot_sample), 0.3, 0.95, rel_tol = 1e-4, show_trace = false, show_every = 1)
  βhat = Optim.minimizer(opt_res_beta)[1]

  # transform to pseudo uniform
  boot_sample_U = mapslices(r -> invperm(sortperm(r, rev=false)), boot_sample; dims = 1) ./ (size(boot_sample, 1) + 1)
  c = 2*quadgk(x -> df(x, βhat, dimension), 0, Inf; atol = 2e-3)[1] # constant
  
  # transform data from u(0,1) to mepd scale
  # boot_sample_exc = zeros(size(vec(boot_sample_U[exc_ind,:]), 1))
  # Threads.@threads for i in 1:size(vec(boot_sample_U[exc_ind,:]), 1)
  #   if i % 100 == 0 
  #     println(i)
  #   end
  #   boot_sample_exc[i] = qF(vec(boot_sample_U[exc_ind,:])[i], βhat, dimension, 1/c; intval = 40) 
  # end
  # boot_sample_exc = reshape(boot_sample_exc, size(boot_sample[exc_ind,:]))
  boot_sample_exc = mapslices(x -> qF.(x, βhat, dimension, 1/c; intval = 40), boot_sample_U[exc_ind,:]; dims = 1) # tr. 

  # estimate mep model
  opt_res = optimize(x -> loglik_cens(x, βhat, boot_sample, boot_sample_exc, dist, thres; isotropic = iso), [log(100.), 1., 1., 1.], NelderMead(), 
    Optim.Options(g_tol = 1e-2, iterations = 200, show_trace = false, show_every = 1, extended_trace = true))

  boot_par_ests[j,:] = [Optim.minimizer(opt_res)..., βhat]

  resultName = @sprintf("appl_bootstrap_res%03.d.dat", j)
  serialize(resultName, [Optim.minimizer(opt_res), βhat])

  println("iteration: ", j, " end, ", "time: ", Dates.now())
end

cd("C:\\Users\\aleen962\\Dropbox\\PhD\\Forskning\\Power exponential dist extremes\\application")
CSV.write("appl_bootstrap_aniso.csv", Tables.table(boot_par_ests))






