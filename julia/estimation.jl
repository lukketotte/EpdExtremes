using Distributed, SharedArrays

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

@everywhere function exceedance_prob(nSims::Int, thres::AbstractVector{<:Real}, cor_mat::AbstractMatrix{<:Real}, β::Real)
    sim = repd(nSims, MvEpd(β, cor_mat))
    return length([i for i in 1:nSims if any(sim[i, :] .> thres)]) / nSims
end

@everywhere function loglik_cens(θ::AbstractVector{<:Real}, β::Real, data::AbstractMatrix{<:Real}, 
    dist::AbstractMatrix{<:Real}, thres::AbstractVector{<:Real})

    if !cond_cor(θ) # check conditions on parameters
      return 1e+10
    end

    cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), size(data, 2), size(data, 2)), θ)
    if !isposdef(cor_mat)
      return 1e+10
    end

    ex_prob = exceedance_prob(10^4, thres, cor_mat, β)
    exc_ind = [i for i in 1:size(data, 1) if any(data[i, :] .> thres)]
    return -(log(1 - ex_prob) * (size(data, 1) - length(exc_ind)) + sum(logpdf(MvEpd(β, cor_mat), permutedims(data[exc_ind,:]))))
end

## Huser procedure, assumes data is on the uniform scale
@everywhere function loglikhuser_cens(θ::AbstractVector{<:Real}, data::AbstractMatrix{<:Real}, 
  dist::AbstractMatrix{<:Real}, thres::Real)

    if !cond_cor(θ) # check conditions on parameters
      return 1e+10
    end
  
    cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), size(data, 2), size(data, 2)), θ)
    if !isposdef(cor_mat)
      return 1e+10
    end
  
  thres_U = quantile.(eachcol(data), thres)
  tresh = first(qG1H(thres_U, θ[3:4]))
  data = qG1H(data, θ[3:4])
  ex_prob = exceedance_prob(10^4, repeat([tresh], size(cor_mat, 1)), cor_mat, θ[3:4])
  exc_ind = [i for i in 1:size(data, 1) if any(data[i, :] .> tresh)]

  return -(log(1 - ex_prob) * (size(data, 1) - length(exc_ind)) + sum(log.(dGH(data[exc_ind,:], cor_mat, θ[3:4])))) + sum(log.(dG1H(data[exc_ind,:], θ[3:4])))
end

@everywhere function exceedance_prob(nSims::Int, thres::AbstractVector{<:Real}, cor_mat::AbstractMatrix{<:Real}, β::AbstractVector{<:Real})
  sim = rGH(nSims, cor_mat, β)
  return length([i for i in 1:nSims if any(sim[i, :] .> thres)]) / nSims
end


####################
####################
####################


λ, ν, β = 1.0, 1.0, 0.75;
true_par = [log(λ), ν, β];
thres = 0.95;
dimension = 5
coord = rand(dimension, 2);
dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]));
cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), dimension, dimension), true_par);
d = MvEpd(β, cor_mat);


reps = 5
mepd = SharedArray{Float64}(reps, 4)
huser = SharedArray{Float64}(reps, 5)
nObs = 500
@sync @distributed for i in 1:reps
  println(i)
  #dat = repd(nObs, d)
  dat = rGH(nObs, cor_mat, [1., 1.])
  ## EPD
  opt_res = optimize(x -> dfmarg(x, dat), [0.75], NelderMead(),
    Optim.Options(g_tol=5e-2, show_trace = true, show_every = 5, extended_trace = true))
  
  βhat = Optim.minimizer(opt_res)[1]

  data_U = mapslices(r -> invperm(sortperm(r, rev=false)), dat; dims = 1) ./ (nObs+1) # data transformed to (pseudo)uniform(0,1)
  thres_U = quantile.(eachcol(data_U), thres) # thresholds on uniform scale
  c = 2*quadgk(x -> df(x, βhat, dimension), 0, Inf; atol = 2e-3)[1] # constant
  data = mapslices(x -> qF.(x, βhat, dimension, 1/c; intval = 20), data_U; dims = 1) # tr
  thresh = repeat([qF(thres_U[1], βhat, dimension, 1/c; intval = 20)], dimension)
  exc_ind = [i for i in 1:size(data, 1) if any(data[i, :] .> thresh)]
  opt_res = optimize(x -> loglik_cens(x, βhat, data, dist, thresh), [log(1.), 1.], NelderMead(),
    Optim.Options(g_tol=9e-2, iterations = 200, show_trace = true, show_every = 20, extended_trace = true))
  aic_mepd = 2*(3 + (loglik_cens(Optim.minimizer(opt_res), βhat, data, dist, thresh) -dfmarg([βhat], data[exc_ind, :])))
  mepd[i,:] = [Optim.minimizer(opt_res)..., βhat, aic_mepd]
  
  ## Huser
  try
    opt_res = optimize(x -> loglikhuser_cens(x, data_U, dist, 0.95), [log(1.0), 1.0, 1., 1.], NelderMead(), 
      Optim.Options(iterations = 200, g_tol = 9e-2, 
      show_trace = true, show_every = 1, extended_trace = true)) 
    aic_huser = 2*(4 + loglikhuser_cens(Optim.minimizer(opt_res), data_U, dist, 0.95))
    huser[i,:] = [Optim.minimizer(opt_res)..., aic_huser]
  catch e
    println(e)
  end
end

mean(mepd, dims = 1)
mean(huser, dims = 1)

mepd
huser