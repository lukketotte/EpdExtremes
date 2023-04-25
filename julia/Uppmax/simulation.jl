using Distributed, SharedArrays, JLD2

@everywhere using Optim, Compat, LinearAlgebra, Statistics, Random, Dates, Distributions, QuadGK, Roots
@everywhere include("../utils.jl")
@everywhere include("../Distributions/mepd.jl")
@everywhere include("../Huser/FFThuser.jl")
@everywhere using .MultivariateEpd, .Utils, .HuserCopula

@everywhere function loglik_cens(θ::AbstractVector{<:Real}, data::AbstractMatrix{<:Real}, dist::AbstractMatrix{<:Real}, thres::AbstractVector{<:Real})

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

@everywhere function loglikhuser_cens(θ::AbstractVector{<:Real}, data::AbstractMatrix{<:Real}, dist::AbstractMatrix{<:Real}, thres::AbstractVector{<:Real})

    if !cond_cor(θ) # check conditions on parameters
      return 1e+10
    end
  
    cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), size(data, 2), size(data, 2)), θ)
    if !isposdef(cor_mat)
      return 1e+10
    end
    
    exc_ind = [i for i in 1:size(data, 1) if any(data[i, :] .> thres)]
    ex_prob = exceedance_prob(10^6, thres, cor_mat, θ[3:4])
    return -((1 - ex_prob) * (size(data, 1) - length(exc_ind)) + sum(log.(dGH(data, cor_mat, θ[3:4]))))
end

@everywhere function exceedance_prob(nSims::Int, thres::AbstractVector{<:Real}, cor_mat::AbstractMatrix{<:Real}, β::Real)
    sim = repd(nSims, MvEpd(β, cor_mat))
    return length([i for i in 1:nObs if any(sim[i, :] .> thres)]) / nSims
end

@everywhere function exceedance_prob(nSims::Int, thres::AbstractVector{<:Real}, cor_mat::AbstractMatrix{<:Real}, β::AbstractVector{<:Real})
    sim = rGH(nSims, cor_mat, β)
    return length([i for i in 1:nObs if any(sim[i, :] .> thres)]) / nSims
end


dimension = 5
nObs = 50

λ = 1.0
ν = 1.0
β = 0.5
true_par = [log(λ), ν, β]

thres = 0.95
coord = rand(dimension, 2)
dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), dimension, dimension), true_par)
#d = MvEpd(β, cor_mat);
d = MvTDist(2, cor_mat)

#dat = repd(nObs, d)
dat = permutedims(rand(d, nObs))
thresh = quantile.(eachcol(dat), thres)

opt_res = optimize(x -> loglik_cens(x, dat, dist, thresh), [log(1.0), 1.0, 0.5], NelderMead(), 
    Optim.Options(g_tol = 1e-2, show_trace = true, show_every = 5, extended_trace = true))
                            
λ_1 = exp(Optim.minimizer(opt_res)[1])
ν_1 = Optim.minimizer(opt_res)[2]
β_1 = Optim.minimizer(opt_res)[3]

opt_res = optimize(x -> loglikhuser_cens(x, dat, dist, thresh), [log(1.0), 1.0, 1., 1.], NelderMead(), 
    Optim.Options(g_tol = 1e-2, show_trace = true, show_every = 5, extended_trace = true))
                            
λ_2 = exp(Optim.minimizer(opt_res)[1])
ν_2 = Optim.minimizer(opt_res)[2]
θ_2 = Optim.minimizer(opt_res)[3:4]

# AIC
2*(3 + loglik_cens([log(λ_1), ν_1, β_1], dat, dist, thresh))
2*(8 + loglikhuser_cens([log(λ_2), ν_2, θ_2...], dat, dist, thresh))

reps = 4*10
mepd = SharedArray{Float64}(reps, 4)
huser = SharedArray{Float64}(reps, 5)

@sync @distributed for i in 1:reps
    dat = permutedims(rand(d, nObs))
    thresh = quantile.(eachcol(dat), thres)
    println("iter $(i)")
    # MEPD
    opt_res = optimize(x -> loglik_cens(x, dat, dist, thresh), [log(1.0), 1.0, 0.5], NelderMead(), 
    Optim.Options(g_tol = 1e-2, show_trace = false, show_every = 5, extended_trace = true))                  
    λ = exp(Optim.minimizer(opt_res)[1])
    ν = Optim.minimizer(opt_res)[2]
    β = Optim.minimizer(opt_res)[3]
    AIC = 2*(3 + loglik_cens([log(λ_1), ν_1, β_1], dat, dist, thresh))
    mepd[i,:] = [λ, ν, β, AIC]
    
    # Huser
    opt_res = optimize(x -> loglikhuser_cens(x, dat, dist, thresh), [log(1.0), 1.0, 1., 1.], NelderMead(), 
    Optim.Options(g_tol = 1e-2, show_trace = false, show_every = 5, extended_trace = true))                
    λ = exp(Optim.minimizer(opt_res)[1])
    ν = Optim.minimizer(opt_res)[2]
    θ = Optim.minimizer(opt_res)[3:4]
    AIC = 2*(8 + loglikhuser_cens([log(λ_2), ν_2, θ_2...], dat, dist, thresh))
    huser[i,:] = [λ, ν, θ..., AIC]
end

mean(mepd, dims = 1)
mean(huser, dims = 1)

# later
"""
@everywhere f(w::Real, t::Real, β::Real, n::Int) = w^((1-n)/2-1) * (1-w)^((n-1)/2 -1) * exp(-0.5*(t/w)^β);
@everywhere g(t::Real, β::Real, n::Int) = t^((n-1)/2) * quadgk(w -> f(w, t, β, n), 0,1; atol = 2e-3)[1];
@everywhere K(β::Real, n::Int) = n*gamma(n/2)/(π^(n/2)*gamma(1+n/(2*β))*2^(1+n/(2*β)))
@everywhere df(x::Real, β::Real, n::Int) = abs(x) > 1e-10 ? g(x^2, β, n) : g(1e-20, β, n)
@everywhere dF(x::Real, β::Real, n::Int, c::Real) = quadgk(y -> c*df(y, β,n),-Inf,x; atol = 1e-4)[1]

@everywhere qF₁(x::Real, p::Real, β::Real, n::Int, c::Real) = dF(x, β, n, c) - p
@everywhere qF(p::Real, β::Real, n::Int, c::Real; intval = 20) = find_zero(x -> qF₁(x, p, β, n, c), (-intval,intval), xatol=2e-3)

c = 2*quadgk(x -> df(x, β, dimension), 0, Inf; atol = 2e-3)[1]
data = mapslices(sortperm, repd(nObs, d); dims = 1) ./ (nObs+1)
exc_ind = [i for i in 1:nObs if any(data[i, :] .> thres)]
U_ep = mapslices(x -> qF.(x, β, dimension, 1/c; intval = 20), data[exc_ind, :]; dims = 1)
U_h = qG1H(data[exc_ind, :], [1.,1])
"""