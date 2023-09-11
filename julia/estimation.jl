using Distributed
addprocs(2)
using SharedArrays, CSV, Random, DelimitedFiles, Tables, Dates

@everywhere using Optim, LinearAlgebra, Distributions, QuadGK, Roots, Printf
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

@everywhere function exceedance_prob(nSims::Int, thres::AbstractVector{<:Real}, cor_mat::AbstractMatrix{<:Real}, β::Real)
  exceedance = 0
  for j in 1:trunc(Int, nSims/1000)
    sim = repd(1000, MvEpd(β, cor_mat))
    exceedance += length([i for i in 1:size(sim, 1) if any(sim[i, :] .> thres)])
  end
  return exceedance / nSims

  ## faster but allocates a lot of memory
  # sim = repd(nSims, MvEpd(β, cor_mat))
  # return length([i for i in 1:nSims if any(sim[i, :] .> thres)]) / nSims
end

@everywhere function loglik_cens(θ::AbstractVector{<:Real}, β::Real, data::AbstractMatrix{<:Real}, data_exc::AbstractMatrix{<:Real}, 
    dist::AbstractMatrix{<:Real}, thres::AbstractVector{<:Real})

    if !cond_cor(θ) # check conditions on parameters
      return 1e+10
    end

    cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), size(data, 2), size(data, 2)), θ)
    if !isposdef(cor_mat)
      return 1e+10
    end

    ex_prob = exceedance_prob(10^6, thres, cor_mat, β) # CHANGED: number of simulations to 1e6 because correlation parameter estimation didn't work with 1e4

    # exc_ind = [i for i in 1:size(data, 1) if any(data[i, :] .> thres)]
    # return -(log(1 - ex_prob) * (size(data, 1) - length(exc_ind)) + sum(logpdf(MvEpd(β, cor_mat), permutedims(data[exc_ind,:]))))
    return -(log(1 - ex_prob) * (size(data, 1) - size(data_exc, 1)) + sum(logpdf(MvEpd(β, cor_mat), permutedims(data_exc))))
end

## Huser procedure, assumes data is on the uniform scale
@everywhere function loglikhuser_cens(θ::AbstractVector{<:Real}, data::AbstractMatrix{<:Real}, 
  dist::AbstractMatrix{<:Real}, thres::Real)

    if !cond_cor_huser(θ) # check conditions on parameters
      return 1e+10
    end
  
    cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), size(data, 2), size(data, 2)), θ)
    if !isposdef(cor_mat)
      return 1e+10
    end
  
  thres_U = quantile.(eachcol(data), thres)
  thresh = first(qG1H(thres_U, θ[3:4]))
  data = qG1H(data, θ[3:4])
  ex_prob = exceedance_prob(10^6, repeat([thresh], size(cor_mat, 1)), cor_mat, θ[3:4])
  exc_ind = [i for i in 1:size(data, 1) if any(data[i, :] .> thresh)]

  return -(log(1 - ex_prob) * (size(data, 1) - length(exc_ind)) + sum(log.(dGH(data[exc_ind,:], cor_mat, θ[3:4])))) + sum(log.(dG1H(data[exc_ind,:], θ[3:4])))
end

@everywhere function exceedance_prob(nSims::Int, thres::AbstractVector{<:Real}, cor_mat::AbstractMatrix{<:Real}, β::AbstractVector{<:Real})
  exceedance = 0
  for j in 1:trunc(Int, nSims/1000)
    sim = rGH(1000, cor_mat, β)
    exceedance += length([i for i in 1:size(sim, 1) if any(sim[i, :] .> thres)])
  end
  return exceedance / nSims
  
  ## faster but allocates a lot of memory
  # sim = rGH(nSims, cor_mat, β)
  # return length([i for i in 1:nSims if any(sim[i, :] .> thres)]) / nSims
end

####################
####################
####################

λ, ν, β = 1.0, 1.0, 0.4;

true_par = [log(λ), ν, β];
thres = 0.95;
dimension = 3;

u2mepd_interval = 70; # intval = ((λ=1: 50; λ=0.5: 70), 20, 17) for β = (0.4, 0.65, 0.9)
trace = true;

reps = 2
mepd = SharedArray{Float64}(reps, 4)
huser = SharedArray{Float64}(reps, 5)
nObs = 100

Random.seed!(123)
@sync @distributed for i in 1:reps
  println(i)

  # CHANGED: each replicate should have both new coordinates and new data
  coord = rand(dimension, 2);
  dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]));
  cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), dimension, dimension), true_par);
  d = MvEpd(β, cor_mat);
  
  data = repd(nObs, d) # generate from mepd
  # data = rGH(nObs, cor_mat, [1., 1.]) # generate from Huser et al model
  
  ## EPD
  # opt_res = optimize(x -> dfmarg(x, data), [0.75], NelderMead(),
  #   Optim.Options(g_tol=1e-5, show_trace = true, show_every = 5, extended_trace = true))
  opt_res = optimize(x -> dfmarg([x], data), 0.3, 0.95, show_trace = trace, show_every = 5) # CHANGED: univariate optimisation of β bcs there was some issue with the multivariate optimisation for β=0.4
  βhat = Optim.minimizer(opt_res)[1]

  data_U = mapslices(r -> invperm(sortperm(r, rev=false)), data; dims = 1) ./ (nObs+1) # data transformed to (pseudo)uniform(0,1)
  thres_U = quantile.(eachcol(data_U), thres) # thresholds on uniform scale
  
  exc_ind = [i for i in 1:size(data_U, 1) if any(data_U[i, :] .> thres_U)]
  c = 2*quadgk(x -> df(x, βhat, dimension), 0, Inf; atol = 2e-3)[1] # constant

  data_exc = mapslices(x -> qF.(x, βhat, dimension, 1/c; intval = u2mepd_interval), data_U[exc_ind,:]; dims = 1) # tr. 
  thresh = repeat([qF(thres_U[1], βhat, dimension, 1/c; intval = u2mepd_interval)], dimension)
  # exc_ind = [i for i in 1:size(data, 1) if any(data[i, :] .> thresh)]

  opt_res = optimize(x -> loglik_cens(x, βhat, data, data_exc, dist, thresh), [log(1.), 1.], NelderMead(), 
    Optim.Options(g_tol = 1e-2, iterations = 100, show_trace = trace, show_every = 20, extended_trace = true)) # CHANGED: g_tol to 1e-2 and iterations to 100 bcs of increased exceedance_prob simulations
  aic_mepd = 2*(3 + (loglik_cens(Optim.minimizer(opt_res), βhat, data, data_exc, dist, thresh) -dfmarg([βhat], data[exc_ind, :])))
  mepd[i,:] = [Optim.minimizer(opt_res)..., βhat, aic_mepd]
  
  resultName = @sprintf("mepd_test%03.d.csv", i)
  CSV.write(resultName, Tables.table([Optim.minimizer(opt_res)..., βhat, aic_mepd]'))

  ## Huser
  # try
    opt_res = optimize(x -> loglikhuser_cens(x, data_U, dist, 0.95), [log(1.0), 1.0, 1., 1.], NelderMead(), 
      Optim.Options(iterations = 100, g_tol = 1e-2, show_trace = trace, show_every = 20, extended_trace = true)) 
    aic_huser = 2*(4 + loglikhuser_cens(Optim.minimizer(opt_res), data_U, dist, 0.95))
    huser[i,:] = [Optim.minimizer(opt_res)..., aic_huser]
  # catch e
  #   println(e)
  # end
  resultName = @sprintf("huser_test%03.d.csv", i)
  CSV.write(resultName, Tables.table([Optim.minimizer(opt_res)..., aic_huser]'))
end

mean(mepd[:,1] .== 0.0)

CSV.write("mepd_$(dimension)_$(month(Dates.today()))_$(day(Dates.today())).csv", Tables.table(mepd))
CSV.write("huser_$(dimension)_$(month(Dates.today()))_$(day(Dates.today())).csv", Tables.table(huser))

# CSV.write("mepd_d10_n200_beta04_la1_nu1_mepd.csv", Tables.table(mepd))
# CSV.write("huser_d10_n200_beta04_la1_nu1_mepd.csv", Tables.table(huser))


