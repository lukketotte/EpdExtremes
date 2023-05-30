module HuserCopula

export rCH, dCH, rGH, rG1, dGH, qG1H, dG1H

using SpecialFunctions, LinearAlgebra, QuadGK, Roots, Distributions, StatsFuns, MvNormalCDF, Random, InvertedIndices

function pFH(r::AbstractVector{<:Real}, par::AbstractVector{<:Real}; log::Bool = false)
  β,γ = par
  res = zeros(Float64, length(r))
  for i in eachindex(res)
    if r[i] > 1
      if β == 0.
        res[i] = log ? log(1 - r[i]^(-γ)) : (1 - r[i]^(-γ))
      else
        ret = 1 - exp(-γ*(r[i]^β - 1)/β)
        res[i] = log ? log(ret) : ret
      end
    end
  end
  res
end

pFH(r::Real, par::AbstractVector{<:Real}; log::Bool = false) = pFH([r], par; log=log)

function qFH(p::AbstractVector{<:Real}, par::AbstractVector{<:Real}; logg::Bool = false)
  all(p .>= 0) && all(p .<= 1) || throw(DomainError(p, " ∉ (0,1)")) 
  β,γ = par
  res = zeros(Float64, length(p))
  for i in eachindex(res)
    if β == 0.
      res[i] = logg ? -log(1 - p[i])/γ : (1 - p[i])^(-1/γ)
    else
      ret = log(1-β*log(1-p[i])/γ)/β
      res[i] = logg ? ret : exp(ret)
    end
  end
  res
end

qFH(p::Real, par::AbstractVector{<:Real}; logg::Bool = false) = qFH([p], par; logg=logg)[1]

function dFH(r::AbstractVector{<:Real}, par::AbstractVector{<:Real}; logg::Bool = false)
  β,γ = par
  res = zeros(Float64, length(r))
  for i in eachindex(res)
    if r[i] > 1
      if β == 0.
        res[i] = log(γ) - (γ + 1)*log(r[i])
      else
        res[i] = -γ*(r[i]^β-1)/β+log(γ)+(β-1)*log(r[i])
      end
    else res[i] = -Inf
    end
  end
  logg ? res : exp.(res)
end

dFH(r::Real, par::AbstractVector{<:Real}; logg::Bool = false) = dFH([r], par; logg = logg)[1]

rFH(n::Integer, par::AbstractVector{<:Real}) = qFH(rand(n),par)

#####################################
## Univariate mixture distriubtion ##
#####################################

# Marginal distribution function (CDF)
pG1const = function (x::Matrix{Float64}, par::AbstractVector{<:Real})
  (n, D) = size(x)
  val = zeros(Float64, (n, D))
  for i in 1:n
    for j in 1:D
      xi = x[i, j]
      if !ismissing(xi)
        val[i, j] = quadgk(x -> pG1_fun(x, xi, par), 0, 1; atol = 2e-3)[1]
      end
    end
  end
  return val
end

pG1_fun = function(prob::Real, x::Real, par::AbstractVector{<:Real})
  return StatsFuns.normcdf(0.0, 1.0, sign(x) * exp(log(abs(x)) - qFH(prob, par; logg = true)))
end

pG1H(x::Matrix{Float64}, par::AbstractVector{<:Real}) = pG1const(x,par)
pG1H(x::Vector{Float64}, par::AbstractVector{<:Real}) = pG1const(reshape(x, (1,length(x))),par)

qG1const = function(prob::Matrix{Float64}, par::AbstractVector{<:Real})
  (n, D) = size(prob)
  val = Matrix{Float64}(undef, n, D)
  for i in 1:n
    for j in 1:D
      prob_i = prob[i, j]
      if !ismissing(prob_i)
        if prob_i <= 0
          val[i, j] = -Inf
        else
          if prob_i >= 1
            val[i, j] = Inf
          else
            try
              val[i, j] = find_zero(x -> qG1_fun(x, prob_i, par), (-20, 20))
            catch e
              #println(par)
              val[i, j] = find_zero(x -> qG1_fun(x, prob_i, par), (-200, 200))
            end
          end
        end
      end
    end
  end
  return val
end

qG1H(prob::Matrix{Float64}, par::AbstractVector{<:Real}) = qG1const(prob,par)
qG1H(prob::Vector{Float64}, par::AbstractVector{<:Real}) = qG1const(reshape(prob, (1,length(prob))),par)

qG1_fun = function(x::Real, prob::Real, par::AbstractVector{<:Real})
  x_mat = reshape([x], 1, 1)
  return pG1H(x_mat, par) .- prob
end

dG1const = function(x::Matrix{Float64}, par::AbstractVector{<:Real})
  (n, D) = size(x)
  val = zeros(Float64, (n, D))
  for i in 1:n
    for j in 1:D
      xi = x[i, j]
      if !ismissing(xi)
        val[i, j] = quadgk(x -> dG1_fun(x, xi, par), 1e-6, 1; atol = 2e-3)[1]
      end
    end
  end
  return val
end

dG1H(x::Matrix{Float64}, par::AbstractVector{<:Real}) = dG1const(x,par)
dG1H(x::Vector{Float64}, par::AbstractVector{<:Real}) = dG1const(reshape(x, (1,length(x))),par)

dG1_fun = function(prob::Real, x::Real, par::AbstractVector{<:Real})
  log_qF = qFH(prob, par; logg = true)
  return exp(logpdf(Normal(), sign(x) * exp(log(abs(x)) - log_qF)) - log_qF)
end
##

# Random generator from marginal distribution G1
rG1 = function(n::Integer, par::AbstractVector{<:Real})
  R = rF(n, par)
  W = rand(Normal(), n)
  return R .* W
end

#################################################################################################################################################
### Multivariate Distribution function, Partial derivatives, Density Function, and Random generator for the Gaussian scale mixture model X=RW ###
#################################################################################################################################################

# Multivariate distribution function (CDF)

pGconst = function(x::Matrix{Float64}, Sigma::Matrix{Float64}, par::AbstractVector{<:Real}) ### x is an nxD matrix; if x is a vector, it is interpreted as a single D-variate vector (not D independent univariate random variables)
  (n, D) = size(x)
  val = zeros(Float64,n)
  for i in 1:n
    val[i] = pGi(x[i, :], Sigma, par)[1]
  end
  return val
end

pGH(x::Matrix{Float64}, Sigma::Matrix{Float64}, par::AbstractVector{<:Real}) = pGconst(x, Sigma, par)
pGH(x::Vector{Float64}, Sigma::Matrix{Float64}, par::AbstractVector{<:Real}) = pGconst(reshape(x, (1, size(Sigma,1))), Sigma, par)

pGi = function(xi::Vector{Float64}, Sigma::Matrix{Float64}, par::AbstractVector{<:Real})
  ind_nna = Vector{Int64}(undef, length(xi))
  num_nna = 0
  for i in eachindex(xi) # this should probably be rewritten without loop
    if !ismissing(xi[i])
      ind_nna[i] = i
      num_nna = num_nna + 1
    else
      ind_nna[i] = missing # check how missing values will be recorded in the data of interest
    end
  end
  return quadgk(x -> pGi_fun(x, xi, Sigma, par, ind_nna), 0, 1; atol = 1e-4)[1]
end

pGi_fun = function(prob::Real, xi::Vector{Float64}, Sigma::Matrix{Float64}, par::AbstractVector{<:Real}, ind_nna::Vector{Int64})
  X = sign.(xi[ind_nna]) .* exp.(log.(abs.(xi[ind_nna])) .- qFH(prob, par; logg = true))
  if length(X) == 1
    val = StatsFuns.normcdf(0.0, Sigma[1], X[1])
  else
    val = mvnormcdf(MvNormal(Sigma[ind_nna, ind_nna]), repeat([-Inf], length(X)), X)[1]
  end
  return val
end

# Multivariate density function (PDF)
dGconst = function(x::Matrix{Float64}, Sigma::Matrix{Float64}, par::AbstractVector{<:Real})
  (n, D) = size(x)
  val = zeros(Float64, n)
  for i in 1:n
    val[i] = dGi(x[i, :], Sigma, par)[1]
  end
  return val
end

dGH(x::Matrix{Float64}, Sigma::Matrix{Float64}, par::AbstractVector{<:Real}) = dGconst(x, Sigma, par)
dGH(x::Vector{Float64}, Sigma::Matrix{Float64}, par::AbstractVector{<:Real}) = dGconst(reshape(x, (1,size(Sigma,1))), Sigma, par)

dGi = function(xi::Vector{Float64}, Sigma::Matrix{Float64}, par::AbstractVector{<:Real})
  ind_nna = Vector{Int64}(undef, length(xi))
  num_nna = 0
  for i in eachindex(xi) # this should probably be rewritten without loop
    if !ismissing(xi[i])
      ind_nna[i] = i
      num_nna = num_nna + 1
    else
      ind_nna[i] = missing # check how missing values will be recorded in the data of interest
    end
  end
  res = 0
  try
    res = quadgk(x -> dGi_fun(x, xi, par, Sigma, ind_nna, num_nna), 1e-8, 1; atol = 2e-3)[1]
  catch e
    res = quadgk(x -> dGi_fun(x, xi, par, Sigma, ind_nna, num_nna), 1e-4, 1; atol = 2e-3)[1]
  end
  res
end

dGi_fun = function(prob::Real, xi::Vector{Float64}, par::AbstractVector{<:Real}, Sigma::Matrix{Float64}, ind_nna::Vector{Int64}, num_nna::Integer)
  log_qF = qFH(prob, par;logg=true)
  X = sign.(xi[ind_nna]) .* exp.(log.(abs.(xi[ind_nna])) .- log_qF)
  if length(X) == 1
    val = exp(logpdf(Normal(Sigma[1]), X[1]) .- num_nna * log_qF)
  else
    val = exp(logpdf(MvNormal(Sigma[ind_nna, ind_nna]), X) .- num_nna * log_qF)
  end
  return val
end
##

rGH = function(n::Integer, Sigma::Matrix{Float64}, par::AbstractVector{<:Real})
  R = rFH(n, par)
  W = rand(MvNormal(Sigma), n)
  return R .* permutedims(W)
end

###################################################################################################################################
### Copula, Partial derivatives, Copula Density Function, and Copula random generator for the Gaussian scale mixture model X=RW ###
###################################################################################################################################

pCconst = function(u::Matrix{Float64}, Sigma::Matrix{Float64}, par::AbstractVector{<:Real}) # u should be a vector of U(0,1)
  return pG(qG1H(u, par), Sigma, par)
end

pC(u::Matrix{Float64}, Sigma::Matrix{Float64}, par::AbstractVector{<:Real}) = pCconst(u,Sigma,par)
pC(u::Vector{Float64}, Sigma::Matrix{Float64}, par::AbstractVector{<:Real}) = pCconst(reshape(u, (1,size(Sigma,1))),Sigma,par)

# Copula density (PDF)
dCconst = function(u::Matrix{Float64}, Sigma::Matrix{Float64}, par::AbstractVector{<:Real})
  qG1_val = qG1H(u, par)
  return log.(dGH(qG1_val, Sigma, par)) .- sum(log.(dG1H(qG1_val, par)), dims = 2)
end

dCH(u::Matrix{Float64}, Sigma::Matrix{Float64}, par::AbstractVector{<:Real}) = dCconst(u,Sigma,par)
dCH(u::Vector{Float64}, Sigma::Matrix{Float64}, par::AbstractVector{<:Real}) = dCconst(reshape(u, (1,size(Sigma,1))),Sigma,par)

function rCH(n::Integer, Sigma::Matrix{Float64},  par::AbstractVector{<:Real})
  return pG1H(rGH(n, Sigma, par), par)
end

# OBS: ensures  precompilation!
dGH([.23, .8], [1 0.2; 0.2 1], [0., 1.5]);
pGH([.23, .8], [1 0.2; 0.2 1], [0., 1.5]);

end