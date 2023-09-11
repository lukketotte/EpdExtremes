using Distributed
# addprocs(4)
using SharedArrays, Random, PositiveFactorizations, PDMats, Plots, KernelDensity, DataFrames
include("./Distributions/alphaStable.jl"); using .AlphaStableDistribution

@everywhere using Optim, LinearAlgebra, Distributions, QuadGK, Roots, Printf, DelimitedFiles, Dates, Tables, CSV, SpecialFunctions, MvNormalCDF
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

@everywhere function dfmarg(β::AbstractVector{<:Real}, data::AbstractMatrix{<:Real}; dat::Bool)  
  if β[1] < 0.95 && β[1] > 0.3
    c = 2*quadgk(x -> df(x, β[1], size(data,2)), 0, Inf; atol = 1e-3)[1]
    # c = 2*quadgk(x -> df(x, β[1], 12), 0, Inf; atol = 1e-3)[1]
    if dat
      return log.((df.(vec(data), β[1], size(data, 2))) ./ c)
    else
      return -sum(log.((df.(vec(data), β[1], size(data, 2))) ./ c))
      # return -sum(log.((df.(data, β[1], 12)) ./ c))
    end
  else
    return 1e+10
  end
end

@everywhere function marg_fun(β) return dfmarg(β, sum(data, dims = 2); dat = false) end # enabling univariate optimisation of β

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

# @everywhere function loglik_cens(θ::AbstractVector{<:Real}, β::Real, data::AbstractMatrix{<:Real}, 
#     data_exc::AbstractMatrix{<:Real}, dist::AbstractMatrix{<:Real}, thres::Real; isotropic::Bool)

#     if !cond_cor(θ) # check conditions on parameters
#       return 1e+10
#     end

#     if isotropic
#       dists_euclid = sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2)
#       dists = reshape(dists_euclid, size(data, 2), size(data, 2))
#     else
#       angle_mat = reshape([cos(θ[3]), sin(θ[3]), -sin(θ[3]), cos(θ[3])], 2, 2)
#       Ω = angle_mat * reshape([1, 0, 0, θ[4]^(-2)], 2, 2) * transpose(angle_mat)
#       dists_mahal = zeros(size(dist, 2))
#       for i in eachindex(dist[1,:])
#           dists_mahal[i] = transpose(dist[:,i]) * inv(Ω) * dist[:,i]
#       end
#       dists = reshape(dists_mahal, size(data, 2), size(data, 2))
#     end
    
#     cor_mat = cor_fun(dists, θ)
#     if !isposdef(cor_mat)
#       return 1e+10
#     end

#     ex_prob = exceedance_prob(trunc(Int, 1e7), thres, cor_mat, β)
    
#     return -(log(1 - ex_prob) * (size(data, 1) - size(data_exc, 1)) + sum(logpdf(MvEpd(β, cor_mat), permutedims(data_exc))))
# end

@everywhere function loglik_cens(θ::AbstractVector{<:Real}, β::Real, data::AbstractMatrix{<:Real}, data_exc::AbstractMatrix{<:Real}, data_marg::AbstractMatrix{<:Real}, dist::AbstractMatrix{<:Real}, thres::Real; isotropic::Bool)

  if !cond_cor(θ) # check conditions on parameters
    return 1e+10
  end

  if isotropic
    dists_euclid = sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2)
    dists = reshape(dists_euclid, size(data, 2), size(data, 2))
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

  ex_prob = exceedance_prob(trunc(Int, 1e7), thres, cor_mat, β)
  
  return -(log(1 - ex_prob) * (size(data, 1) - size(data_exc, 1)) + sum(logpdf(MvEpd(β, cor_mat), permutedims(data_exc)) .- sum(data_marg, dims = 2)))
end


## Gaussian model
@everywhere function loglik_gaussian_cens(θ::AbstractVector{<:Real}, data::AbstractMatrix{<:Real}, dist::AbstractMatrix{<:Real}, thres::Real; isotropic::Bool)

  if !cond_cor(θ) # check conditions on parameters
    return 1e+10
  end

  if isotropic
    dists_euclid = sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2)
    dists = reshape(dists_euclid, size(data, 2), size(data, 2))
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

  thres_U = quantile.(eachcol(data), thres)
  thresh = first(quantile(Normal(), thres_U))
  data_gauss = quantile(Normal(), data)
  ex_prob = exceedance_prob_gauss(trunc(Int, 1e7), thresh, cor_mat)

  exc_ind = [i for i in 1:size(data_gauss, 1) if any(data_gauss[i, :] .> thresh)]
  # return -(log(1 - ex_prob) * (size(data, 1) - length(exc_ind)) + sum(logpdf(MvNormal(cor_mat), permutedims(data[exc_ind,:]))))
  return -(log(1 - ex_prob) * (size(data, 1) - length(exc_ind)) + sum(logpdf(MvNormal(cor_mat), permutedims(data_gauss[exc_ind,:])) .- sum(logpdf.(Normal(), data_gauss[exc_ind,:]), dims = 2)))
end

@everywhere function exceedance_prob_gauss(nSims::Int, thres::Real, cor_mat::AbstractMatrix{<:Real})
  exceedance = 0
  for j in 1:trunc(Int, nSims/10000)
    sim = rand(MvNormal(cor_mat), 10000)
    exceedance += length([i for i in 1:size(sim, 1) if any(sim[i, :] .> thres)])
  end
  return exceedance / nSims

  ## faster but allocates a lot of memory
  # sim = permutedims(rand(MvNormal(cor_mat), nSims))
  # return length([i for i in 1:nSims if any(sim[i, :] .> thres)]) / nSims
end



####################
####################
####################

# pwd()
cd("c:/Users/aleen962/Dropbox/PhD/Forskning/Power exponential dist extremes/application/data/data_sets")
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
@time opt_res = optimize(x -> dfmarg([x], sum(data, dims = 2); dat=false), 0.3, 0.95, rel_tol = 1e-3, show_trace = true, show_every = 1) # CHANGED: univariate optimisation of β bcs there was some issue with the multivariate optimisation for β=0.4
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
# marg_data = reshape(dfmarg([βhat], data_exc; dat = true), size(data_exc))

# save transformed data
# using Tables
# cd("C:\\Users\\aleen962\\Dropbox\\PhD\\Forskning\\Power exponential dist extremes\\application")
# CSV.write("gustData_mepdTrans_u098.csv", Tables.table(data_exc))
# CSV.write("gustData_marginal_u098.csv", Tables.table(marg_data))
#

### fit mepd likelihood
# load transformed data and βhat
cd("c:/Users/aleen962/Dropbox/PhD/Forskning/Power exponential dist extremes/application")
data_exc = convert(Matrix{Float64}, readdlm("gustData_mepdTrans_u098.csv", ',')[2:end,:])
βhat = readdlm("betaHat.csv", ',')[1]
marg_data = convert(Matrix{Float64}, readdlm("gustData_marginal_u098.csv", ',')[2:end,:])

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
opt_res = optimize(x -> loglik_cens(x, βhat, data, data_exc, marg_data, dist, thres; isotropic=true), [log(100.), 1., 1., 1.], NelderMead(), 
    Optim.Options(g_tol = 1e-2, iterations = 300, show_trace = true, show_every = 10, extended_trace = true))
#
Optim.minimizer(opt_res)[1], Optim.minimizer(opt_res)[2], Optim.minimizer(opt_res)[3], Optim.minimizer(opt_res)[4]

# log-likelihood value
# logL_aniso = -(loglik_cens(Optim.minimizer(opt_res), βhat, data, data_exc, dist, thres; isotropic=false) -dfmarg([βhat], data_exc))
logL_aniso = -(loglik_cens(Optim.minimizer(opt_res), βhat, data, data_exc, marg_data, dist, thres; isotropic=false))
# AIC
# aic_aniso = 2*((length(Optim.minimizer(opt_res))+1) + (loglik_cens(Optim.minimizer(opt_res), βhat, data, data_exc, dist, thres; isotropic=false) -dfmarg([βhat], data_exc)))
aic_aniso = 2*((length(Optim.minimizer(opt_res))+1) - logL_aniso)
#
appl_results = [Optim.minimizer(opt_res)..., βhat, logL_aniso, aic_aniso]
# writedlm("application_results_mepd_anisotropic.csv", appl_results, ',') # save results as .csv



###############################
### Fit Gaussian log-likelihood
###############################

### isotropic correlation model
opt_res_gauss_iso = optimize(x -> loglik_gaussian_cens(x, data_U, dist, 0.95; isotropic=true), [log(100.), 1.], NelderMead(), 
    Optim.Options(iterations = 1000, show_trace = true, show_every = 10, extended_trace = true))
#
Optim.minimizer(opt_res_gauss_iso)[1], Optim.minimizer(opt_res_gauss_iso)[2]
#log-likelihood value
logL_gauss_iso = -(loglik_gaussian_cens(Optim.minimizer(opt_res_gauss_iso), data_U, dist, 0.95; isotropic=true))
# AIC
aic_gauss_iso = 2*((length(Optim.minimizer(opt_res_gauss_iso)) + 1) - logL_gauss_iso)
#
appl_results_iso = [Optim.minimizer(opt_res_gauss_iso)..., logL_gauss_iso, aic_gauss_iso]
# writedlm("application_results_gauss_isotropic.csv", appl_results_iso, ',') # save results as .csv


### anisotropic correlation model
opt_res_gauss_aniso = optimize(x -> loglik_gaussian_cens(x, data_U, dist, 0.95; isotropic=false), [log(100.), 1., 1., 1.], NelderMead(), 
    Optim.Options(iterations = 1000, show_trace = true, show_every = 10, extended_trace = true))
#
Optim.minimizer(opt_res_gauss_aniso)[1], Optim.minimizer(opt_res_gauss_aniso)[2], Optim.minimizer(opt_res_gauss_aniso)[3], Optim.minimizer(opt_res_gauss_aniso)[4]
# log-likelihood value
logL_gauss_aniso = -(loglik_gaussian_cens(Optim.minimizer(opt_res_gauss_aniso), data_U, dist, 0.95; isotropic=false))
# AIC
aic_gauss_aniso = 2 * (length(Optim.minimizer(opt_res_gauss_aniso)) - logL_gauss_aniso)
#
appl_results_aniso = [Optim.minimizer(opt_res_gauss_aniso)..., logL_gauss_aniso, aic_gauss_aniso]
# writedlm("application_results_gauss_anisotropic.csv", appl_results_aniso, ',') # save results as .csv




###############################################################################################
##### Diagnostics ##### Diagnostics ##### Diagnostics ##### Diagnostics ##### Diagnostics #####
###############################################################################################

#######################
### Spatial predictions
#######################
cd("c:/Users/aleen962/Dropbox/PhD/Forskning/Power exponential dist extremes/application")
appl_grid_points = convert(Matrix{Float64}, readdlm("appl_grid_points.csv", ',')[2:end,:])
θ = reshape(readdlm("application_results_anisotropic.csv")[1:4,:], 4)
# prediction_dists = readdlm("prediction_dists.csv")
@time prediction_dists = vcat(dist_fun(appl_grid_points[:, 1]), dist_fun(appl_grid_points[:, 2]))

dimension = size(appl_grid_points, 1)
prediction_dists_mat = reshape(sqrt.(prediction_dists[1, :] .^ 2 .+ prediction_dists[2, :] .^ 2), dimension, dimension)

x₁ = data[28,:] # we condition on row 28 in the data matrix
# Σ₁₁ = cor_fun(abs.(vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))), θ)
Σ = cor_fun(prediction_dists_mat, θ)
k = size(Σ,1)
l = length(x₁)

μ = Σ[1:(k-l), (k-l+1):end] * inv(Σ[(k-l+1):end, (k-l+1):end]) * x₁
σ = Σ[1:(k-l), 1:(k-l)] - Σ[1:(k-l), (k-l+1):end] * inv(Σ[(k-l+1):end, (k-l+1):end]) * Σ[1:(k-l), (k-l+1):end]'

if !isposdef(σ)
  σ_pos = PDMat(σ, cholesky(Positive, σ))
end

# Random.seed!(123)
v = rmix(1, x₁, inv(Σ[(k-l+1):end, (k-l+1):end]), βhat)
appl_spatial_preds = v.*rand(MvNormal(μ./v, σ_pos))
# CSV.write("appl_spatial_preds.csv", Tables.table(appl_spatial_preds))


###############################
### compute χ for fitted models
###############################

## calculate correlations matrix based on parameter estimates
using Cubature, StatsPlots
@everywhere g(x::Float64, p::Real) = exp(-abs(x)^p/2) / (π * gamma(1+1/p) * 2^(1/p))
@everywhere f(y::Float64, x::Float64, p::Real) = (y-x^2 + 1.e-20)^(-1/2) * g(y, p)
@everywhere depd(x::Float64, p::Real) = quadgk(y -> f(y, x, p), x^2, Inf)[1]
@everywhere pepd(x::Float64, p::Real) = 1/2 + quadgk(y -> depd(y, p), 0, x)[1]
# quantile function using roots
@everywhere function qepd(q::Real, p::Real; x0::Real = 1)
  f(x) = pepd(x, p) - q
  find_zero(f, x0, Order16())
end
@everywhere function C(x::Float64, y::Float64, p::Real, ρ::Real)
  1/√(1-ρ^2) * g((x^2 + y^2 - 2*ρ*x*y)/(1-ρ^2), p)
end
@everywhere C1(x::Float64, p::Real, ρ::Real, q::Real) = quadgk(y -> C(y, x, p, ρ), -Inf, q)[1]
@everywhere C2(p::Real, ρ::Real, q::Real) = quadgk(y -> C1(y, p, ρ, q), -Inf, q; rtol=1e-12)[1]

@everywhere function pC_Gauss(u, Σ)
  x = quantile(Normal(), u)
  return(mvnormcdf(MvNormal(Σ), repeat([-Inf], length(x)), x))
end

cd("C:/Users/aleen962/Dropbox/PhD/Forskning/Power exponential dist extremes/application")
coord = convert(Matrix{Float64}, readdlm("./data/data_sets/wind_gust_coordinates_km.csv", ',')[2:end,:]) # lon, lat
data = convert(Matrix{Float64}, readdlm("./data/data_sets/model_data_complete.csv", ',')[2:end,:])
dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))

##### MEPD
## mepd isotropic
# dists_euclid = sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2)
# dists = reshape(dists_euclid, size(data, 2), size(data, 2))
# param_ests = readdlm("application_results_isotropic.csv")[1:3]
# θ, βhat = param_ests[1:2], param_ests[3]
##

# mepd anisotropic
param_ests = readdlm("application_results_anisotropic.csv")[1:5]
θ, βhat = param_ests[1:4], param_ests[5]
angle_mat = reshape([cos(θ[3]), sin(θ[3]), -sin(θ[3]), cos(θ[3])], 2, 2)
Ω = angle_mat * reshape([1, 0, 0, θ[4]^(-2)], 2, 2) * transpose(angle_mat)
dists_mahal = zeros(size(dist, 2))
for i in eachindex(dist[1,:])
    dists_mahal[i] = transpose(dist[:,i]) * inv(Ω) * dist[:,i]
end
dists = reshape(dists_mahal, size(data, 2), size(data, 2))
##

##### GAUSSIAN
## gauss isotropic
# dists_euclid = sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2)
# dists = reshape(dists_euclid, size(data, 2), size(data, 2))
# θ = readdlm("application_results_gauss_isotropic.csv")[1:2]
##

# gauss anisotropic
θ = readdlm("application_results_gauss_anisotropic.csv")[1:4]
angle_mat = reshape([cos(θ[3]), sin(θ[3]), -sin(θ[3]), cos(θ[3])], 2, 2)
Ω = angle_mat * reshape([1, 0, 0, θ[4]^(-2)], 2, 2) * transpose(angle_mat)
dists_mahal = zeros(size(dist, 2))
for i in eachindex(dist[1,:])
    dists_mahal[i] = transpose(dist[:,i]) * inv(Ω) * dist[:,i]
end
dists = reshape(dists_mahal, size(data, 2), size(data, 2))
##

### correlation matrix
cor_mat = cor_fun(dists, θ)

#### Figure of Dependence Measures for Extreme Value Analyses ####
# u = [range(0.9, 0.98, length = 5); range(0.99, 1, length = 6)]
u = [0.9, 0.95, 0.99]
ρ = zeros(66)
ind = 0
for i in 1:11
  for j in (i+1):12
    ind += 1
    ρ[ind] = cor_mat[i,j]
  end
end
###


## mepd
βhat = readdlm("betaHat.csv", ',')[1]
# βhat = 0.95
plot_dat = DataFrame(rho = repeat(ρ, inner = length(u)) |> x -> repeat(x, inner = length([βhat])),
                     thresh = repeat(u, length(ρ)) |> x -> repeat(x, inner = length([βhat])),
                     beta = repeat([βhat], length(ρ)*length(u)), val = 0.)


cols = names(plot_dat)
plot_dat = SharedArray(Matrix(plot_dat))

Dates.now()
@sync @distributed for i ∈ 1:lastindex(plot_dat[:,1])
  println(i)
  if plot_dat[i, 2] != 1.
      a = qepd(plot_dat[i, 2], plot_dat[i, 3])
      val = C2(plot_dat[i, 3], plot_dat[i, 1], a)
      plot_dat[i, 4] = 2 - log(val)/log(plot_dat[i, 2])
  else
      plot_dat[i,4] = 0
  end
end

chi_plot_dat_mepd_aniso = DataFrame(Tables.table(plot_dat)) |> x -> rename!(x, cols)
# CSV.write("chi_plot_dat_mepd_aniso.csv", chi_plot_dat_mepd_aniso)
@df chi_plot_dat_mepd_aniso plot(:thresh, :val, group = :rho, legend=:none)
##



## Gauss
plot_dat_gauss = DataFrame(rho = repeat(ρ, inner = length(u)), thresh = repeat(u, length(ρ)), val = 0.)
cols = names(plot_dat_gauss)
plot_dat_gauss = SharedArray(Matrix(plot_dat_gauss))

Dates.now()
@sync @distributed for i ∈ 1:lastindex(plot_dat_gauss[:,1])
  println(i)
  if plot_dat_gauss[i, 2] != 1.
      val = pC_Gauss([plot_dat_gauss[i, 2], plot_dat_gauss[i, 2]], reshape([1, plot_dat_gauss[i, 1], plot_dat_gauss[i, 1], 1], 2, 2))[1]
      plot_dat_gauss[i, 3] = 2 - log(val)/log(plot_dat_gauss[i, 2])
  else
      plot_dat_gauss[i,3] = 0
  end
end

chi_plot_dat_gauss_aniso = DataFrame(Tables.table(plot_dat_gauss)) |> x -> rename!(x, cols)
# CSV.write("chi_plot_dat_gauss_aniso.csv", chi_plot_dat_gauss_aniso)
@df chi_plot_dat_gauss_aniso plot(:thresh, :val, group = :rho, legend=:none)
##




################################
### QQ-plots for sum exceedances
################################

cd("C:/Users/aleen962/Dropbox/PhD/Forskning/Power exponential dist extremes/application")
coord = convert(Matrix{Float64}, readdlm("./data/data_sets/wind_gust_coordinates_km.csv", ',')[2:end,:]) # lon, lat
dimension = size(coord, 1)
dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]));
data = convert(Matrix{Float64}, readdlm("./data/data_sets/model_data_complete.csv", ',')[2:end,:])
data_U = mapslices(r -> invperm(sortperm(r, rev=false)), data; dims = 1) ./ (size(data, 1) + 1) # data transformed to (pseudo)uniform(0,1)

sum_data_exc = readdlm("gustData_sum_exceed_u95_noHead.csv", ',', Float64)

## locate sum exceedances
sum_data = reshape(sum(data, dims = 2), size(data, 1))
sum_data_exc_inds = [i for i in eachindex(sum_data) if sum_data[i] .> quantile(sum_data, 0.95)]

## transform data to pseudo uniform scale
# c = 2*quadgk(x -> df(x, βhat, dimension), 0, Inf; atol = 2e-3)[1] # constant

### transform sum exceedance data from u(0,1) to mepd scale
# sum_data_exc = zeros(size(vec(data_U[sum_data_exc_inds,:]), 1))
# using Dates; Dates.now()
# @time Threads.@threads for i in 1:size(vec(data_U[sum_data_exc_inds,:]), 1)
#   println(i)
#   sum_data_exc[i] = qF(vec(data_U[sum_data_exc_inds,:])[i], βhat, size(coord, 1), 1/c; intval = 40) 
# end
# sum_data_exc = reshape(sum_data_exc, size(data_U[sum_data_exc_inds,:]))
# CSV.write("gustData_sum_exceed_u95.csv", Tables.table(sum_data_exc))


## calculate correlations matrix based on parameter estimates
# isotropic distances
param_ests = readdlm("application_results_isotropic.csv")[1:3]
θ, βhat = param_ests[1:2], param_ests[3]
dists = reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), size(data, 2), size(data, 2))

# anisotropic distances
param_ests = readdlm("application_results_anisotropic.csv")[1:5]
θ, βhat = param_ests[1:4], param_ests[5]
angle_mat = reshape([cos(θ[3]), sin(θ[3]), -sin(θ[3]), cos(θ[3])], 2, 2)
Ω = angle_mat * reshape([1, 0, 0, θ[4]^(-2)], 2, 2) * transpose(angle_mat)
dists_mahal = zeros(size(dist, 2))
for i in eachindex(dist[1,:])
    dists_mahal[i] = transpose(dist[:,i]) * inv(Ω) * dist[:,i]
end
dists = reshape(dists_mahal, size(data, 2), size(data, 2))

### correlation matrix
cor_mat = cor_fun(dists, θ)
#

### mepd
sum_var = sum(cor_mat)
mepd_sum_exc = sort(sum(sum_data_exc, dims = 2), dims = 1)
quants = range(0.95, 0.9999, length = size(mepd_sum_exc, 1))

# d = MvEpd(βhat, reshape([sum_var], 1, 1))
# sim_data = repd(10^7, d)
d = MvEpd(βhat, cor_mat)
sim_data = sum(repd(10^7, d), dims = 2)
mepd_model_quants = quantile(sim_data, quants)

plot(mepd_sum_exc./sqrt(sum_var), mepd_model_quants./sqrt(sum_var); seriestype=:scatter)


### gaussian
# anisotropic
θ = readdlm("application_results_gauss_anisotropic.csv")[1:4]
angle_mat = reshape([cos(θ[3]), sin(θ[3]), -sin(θ[3]), cos(θ[3])], 2, 2)
Ω = angle_mat * reshape([1, 0, 0, θ[4]^(-2)], 2, 2) * transpose(angle_mat)
dists_mahal = zeros(size(dist, 2))
for i in eachindex(dist[1,:])
    dists_mahal[i] = transpose(dist[:,i]) * inv(Ω) * dist[:,i]
end
dists = reshape(dists_mahal, size(data, 2), size(data, 2))
### correlation matrix
cor_mat_gauss = cor_fun(dists, θ)
#


data_gauss = quantile(Normal(), data_U)
sum_data_gauss = sum(data_gauss, dims = 2)
sum_var_gauss = sum(cor_mat_gauss)
gauss = MvNormal(reshape([sum_var_gauss], 1, 1))

quants = range(0.95, 0.9999, length = size(mepd_sum_exc, 1))

gauss_model_quants = quantile(rand(gauss, 10^7), quants)
gauss_sum_exc = quantile(sum_data_gauss, quants)

plot(gauss_sum_exc./sqrt(sum_var), gauss_model_quants./sqrt(sum_var); seriestype=:scatter)


CSV.write("qq_plot_data.csv", Tables.table(hcat(quants, mepd_sum_exc, mepd_model_quants, gauss_sum_exc, gauss_model_quants)))


############# copy-pasted from mepd.jl
using Distributions, LinearAlgebra, SpecialFunctions, PDMats, Random
import Base: rand, convert, length
import Distributions: pdf, logpdf, @check_args, params, sqmahal

abstract type AbstractMvEpd <: ContinuousMultivariateDistribution end

struct GenericMvEpd{T<:Real, Cov<:AbstractPDMat, Mean<:AbstractVector} <: AbstractMvEpd
    p::T
    dim::Int
    μ::Mean
    Σ::Cov

    function GenericMvEpd{T, Cov, Mean}(p::T, dim::Int, μ::Mean, Σ::AbstractPDMat{T}) where {T, Cov, Mean}
        p > zero(p) || error("p must be positive")
        new{T, Cov, Mean}(p, dim, μ, Σ)
    end
end

function GenericMvEpd(p::T, μ::Mean, Σ::Cov) where {Cov<:AbstractPDMat, Mean<:AbstractVector,T<:Real}
    d = length(μ)
    dim(Σ) == d || throw(DimensionMismatch("The dimensions of μ and Σ are inconsistent"))
    R = Base.promote_eltype(T, μ, Σ)
    S = convert(AbstractArray{R}, Σ)
    m = convert(AbstractArray{R}, μ)
    GenericMvEpd{R, typeof(S), typeof(m)}(R(p), d, m, S)
end

function GenericMvEpd(p::Real, Σ::AbstractPDMat)
    R = Base.promote_eltype(p, Σ)
    GenericMvEpd(p, zeros(R, dim(Σ)), Σ)
end

function convert(::Type{GenericMvEpd{T}}, d::GenericMvEpd) where T <:Real
    S = convert(AbstractArray{T}, d.Σ)
    m = convert(AbstractArray{T}, d.μ)
    GenericMvEpd{T, typeof(S), typeof(m)}(T(d.p), d.dim, m, S)
end
Base.convert(::Type{GenericMvEpd{T}}, d::GenericMvEpd{T}) where {T<:Real} = d

function convert(::Type{GenericMvEpd{T}}, p, dim, μ::AbstractVector, Σ::AbstractPDMat) where T<:Real
    S = convert(AbstractArray{T}, Σ)
    m = convert(AbstractArray{T}, μ)
    GenericMvEpd{T, typeof(S), typeof(m)}(T(p), dim, m, S)
end

MvEpd(p::Real, μ::Vector{<:Real}, Σ::PDMat) = GenericMvEpd(p, μ, Σ)
MvEpd(p::Real, Σ::PDMat) = GenericMvEpd(p, Σ)
MvEpd(p::Real, μ::Vector{<:Real}, Σ::Matrix{<:Real}) = GenericMvEpd(p, μ, PDMat(Σ))
MvEpd(p::Real, Σ::Matrix{<:Real}) = GenericMvEpd(p, PDMat(Σ))

length(d::GenericMvEpd) = d.dim
params(d::GenericMvEpd) = (d.p, d.dim, d.μ, d.Σ)
sqmahal(d::GenericMvEpd, x::AbstractVector{<:Real}) = invquad(d.Σ, x - d.μ)

"""function mvepd_const(d::AbstractMvEpd)
    H = convert(eltype(d), pi^(-d.dim/2))
    H * d.dim*gamma(d.dim/2) / (gamma(1+d.dim/(2*d.p)) * 2^(1+d.dim/(2*d.p)))
end"""

function mvepd_const(d::AbstractMvEpd)
    H = convert(eltype(d), pi^(-d.dim/2))
    log(H) + log(d.dim) + loggamma(d.dim/2) - loggamma(1+d.dim/(2*d.p)) - (1+d.dim/(2*d.p))*log(2)
end

function logpdf(d::AbstractMvEpd, x::AbstractVector{T}) where T<:Real
    k = mvepd_const(d)
    mvepd_const(d) -0.5 * logdet(d.Σ) -0.5*sqmahal(d, x)^d.p
end

pdf(d::AbstractMvEpd, x::AbstractVector{<:Real}) = exp(logpdf(d, x))

function runifsphere(d::Int)
    mvnorm = rand(Normal(), d)
    mvnorm ./ sqrt.(sum(mvnorm.^2))
end

function repd(n::Int, d::GenericMvEpd)
    p, dim, μ, Σ = params(d)
    Σ = sqrt(Σ)
    res = zeros(n,dim)
    for i ∈ 1:n
        R = rand(Gamma(2,12/(2*p))).^(1/(2*p)) # changed dimension to 12
        res[i,:] = μ + R*Σ*runifsphere(dim)
    end
    res
end

d = MvEpd(βhat, reshape([sum_var],1,1))
sim_data = sum(repd(10^6, d), dims = 2)
mepd_model_quants = quantile(sim_data, quants)

plot(mepd_sum_exc, mepd_model_quants; seriestype=:scatter)
