using Distributed, SharedArrays, CSV, Random, DelimitedFiles, Dates, Tables, Printf
# addprocs(6)

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



# load data
coord = convert(Matrix{Float64}, readdlm("wind_gust_coordinates_km.csv", ',')[2:end,:]) # lon, lat
dimension = size(coord, 1)
dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]));
data = convert(Matrix{Float64}, readdlm("model_data_complete.csv", ',')[2:end,:])
thres_q = 0.98

reps = 10
boot_par_ests = SharedArray{Float64}(reps, 3)
Random.seed!(789)
@sync @distributed for j in 1:reps

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
#   boot_sample_exc = zeros(size(vec(boot_sample_U[exc_ind,:]), 1))
#   for i in 1:size(vec(boot_sample_U[exc_ind,:]), 1)
#     boot_sample_exc[i] = qF(vec(boot_sample_U[exc_ind,:])[i], βhat, dimension, 1/c; intval = 40) 
#   end
#   boot_sample_exc = reshape(boot_sample_exc, size(boot_sample[exc_ind,:]))
  boot_sample_exc = mapslices(x -> qF.(x, βhat, dimension, 1/c; intval = 40), boot_sample_U[exc_ind,:]; dims = 1) # tr. 

  # estimate mep model
  opt_res = optimize(x -> loglik_cens(x, βhat, boot_sample, boot_sample_exc, dist, thres; isotropic=false), [log(100.), 1., 1., 1.], NelderMead(), 
    Optim.Options(g_tol = 1e-2, iterations = 200, show_trace = false, show_every = 1, extended_trace = true))

  boot_par_ests[j,:] = [Optim.minimizer(opt_res), βhat]

  resultName = @sprintf("appl_bootstrap_res%03.d.dat", j)
  serialize(resultName, [Optim.minimizer(opt_res), βhat])
  
  println("iteration: ", j, " end, ", "time: ", Dates.now())
end

CSV.write("appl_bootstrap_aniso_$(month(Dates.today()))_$(day(Dates.today())).csv", Tables.table(boot_par_ests))