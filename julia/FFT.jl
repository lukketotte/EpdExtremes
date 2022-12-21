using SpecialFunctions, LinearAlgebra, QuadGK, Roots, Distributions, StatsFuns, MvNormalCDF
include("./Constants/qFinterval.jl")
using .QFinterval
using BenchmarkTools

ζ(α::Real) = -tan(π*α/2)
θ₀(α::Real) = 1/α * atan(tan(π*α/2))
function V(α::Real, θ::Real)
  ϑ = θ₀(α)
  a = cos(α*ϑ)^(1/(α-1))
  b = (cos(θ)/sin(α*(ϑ + θ)))^(α/(α-1))
  c = cos(α*ϑ + (α-1)*θ) / cos(θ)
  return a*b*c 
end

h(θ::Real, x::Real, α::Real) = (x-ζ(α))^(α/(α-1))*V(α,θ)*exp(-(x-ζ(α))^(α/(α-1))*V(α,θ))
f(x::Real, α::Real) = α/(π*(x-ζ(α))*abs(α-1)) * quadgk(θ -> h(θ, x, α), -θ₀(α), α <= 0.95 ? π/2 : 1.57)[1]
dstable(x::Real, α::Real, γ::Real) = f((x-γ * tan(π*α/2))/γ, α)/γ

dF = function(x::Real, p::Real, d::Int)
  p > 0 && p < 1 || throw(DomainError(p,"must be on (0,1)")) 
  γ = 2^(1-1/p) * cos(π*p/2)^(1/p)
  C = 2^(1+d/2*(1-1/p)) * gamma(1+d/2) / gamma(1+d/(2*p))
  x > 0 ? C*x^(d-3)*dstable(x^(-2), p, γ) : 0
end

pF = function(x::Real, p::Real, d::Int)
  quadgk(x -> dF(x,p,d), 0, x)[1]
end

qF₁(x::Real, prob::Real, p::Real, d::Integer) = pF(x, p, d) - prob


qF = function(prob::Real, p::Real, d::Integer)
  prob > 0 && prob < 1 || throw(DomainError(prob, "must be on (0,1)"))
  try
    find_zero(x -> qF₁(x, prob, p, d), getInterval(prob, p, d), xatol=1e-4)
  catch e
    if isa(e, DomainError) || isa(e, ArgumentError)
      find_zero(x -> qF₁(x, prob, p, d), (0.01, 100), xatol=1e-4)
    end
  end
end

rF = function(n::Integer, p::Real, d::Integer)
  ret = zeros(n)
  for i in eachindex(ret)
    ret[i] = qF(rand(Uniform()), p, d)
  end
  ret
end


#####################################
## Univariate mixture distriubtion ##
#####################################

# Marginal distribution function (CDF)
pG1 = function(x, p)
  xmat = x
  if isa(xmat, Vector)
    (n, D) = (size(xmat)[1], 1)
    xmat = reshape(xmat, n, D)
  elseif isa(xmat, Matrix)
    (n, D) = size(xmat)
  else
    xmat = reshape([xmat], 1, 1)
    (n, D) = (1, 1)
  end  
  val = Matrix{Float64}(undef, n, D)
  for i in 1:n
    for j in 1:D
      xi = x[i, j]
      if !ismissing(xi)
        val[i, j] = quadgk(x -> pG1_fun(x, xi, p, D), 0, 1)[1]
      end
    end
  end
  if !isa(x, Matrix)
    val = reshape(val, size(val)[1])
  end
  val
end

pG1_fun = function(prob, x, p, d)
  StatsFuns.normcdf(0.0, 1.0, sign(x) * exp( log(abs(x)) - log(qF(prob, p, d)) ))
end
##

# Marginal quantile function
qG1 = function(prob, p)
  prob_mat = prob
  if isa(prob_mat, Vector)
    (n, D) = (size(prob_mat)[1], 1)
    prob_mat = reshape(prob_mat, n, D)
  elseif isa(prob_mat, Matrix)
    (n, D) = size(prob_mat)
  else
    prob_mat = reshape([prob_mat], 1, 1)
    (n, D) = (1, 1)
  end
  val = Matrix{Float64}(undef, n, D)
  for i in 1:n
    for j in 1:D
      p_i = prob_mat[i, j]
      if !ismissing(p_i)
        if p_i <= 0
          val[i, j] = -Inf
        else
          if p_i >= 1
            val[i, j] = Inf
          else
            val[i, j] = find_zero(x -> qG1_fun(x, p_i, p), (-10^2, 10^2))
          end
        end
      end
    end
  end
  if !isa(prob, Matrix)
    val = reshape(val, size(val)[1])
  end
  val
end

qG1_fun = function(x, prob, p)
  pG1(x, p) .- prob
end
##

# Marginal density function (PDF)
dG1 = function (x, p)
  xmat = x
  if isa(xmat, Vector)
    (n, D) = (size(xmat)[1], 1)
    xmat = reshape(xmat, n, D)
  elseif isa(xmat, Matrix)
    (n, D) = size(xmat)
  else
    xmat = reshape([xmat], 1, 1)
    (n, D) = (1, 1)
  end
  val = Matrix{Float64}(undef, n, D)
  for i in 1:n
    for j in 1:D
      xi = x[i, j]
      if !ismissing(xi)
        val[i, j] = quadgk(x -> dG1_fun(x, xi, p, D), 0, 1)[1]
      end
    end
  end
  if !isa(x, Matrix)
    val = reshape(val, size(val)[1])
  end
  val
end

dG1_fun = function (prob, x, p, d)
  log_qF = log( qF(prob, p, d) )
  exp( StatsFuns.normlogcdf(0.0, 1.0, sign(x) * exp(log(abs(x)) - log_qF) - log_qF) )
end
##

# Random generator from marginal distribution G1
rG1 = function (n, p, d)
  R = rF(n, p, d)
  W = rand(Normal(), n)
  R .* W
end


#################################################################################################################################################
### Multivariate Distribution function, Partial derivatives, Density Function, and Random generator for the Gaussian scale mixture model X=RW ###
#################################################################################################################################################

# Multivariate distribution function (CDF)
pG = function (x, Sigma, p) ### x is an nxD matrix; if x is a vector, it is interpreted as a single D-variate vector (not D independent univariate random variables)
  if isa(x, Vector)
    (n, D) = (size(x)[1], 1)
    x = reshape(x, n, D)
  elseif isa(x, Matrix)
    (n, D) = size(x)
  else
    x = reshape([x], 1, 1)
    (n, D) = (1, 1)
  end
  val = Vector{Float64}(undef, n)
  for i in 1:n
    val[i] = pGi(x[i, :], Sigma, p)
  end
  val
end

pGi = function (xi, Sigma, p)
  ind_nna = Vector{Int64}(undef, length(xi)) # this should probably be rewritten without loop
  num_nna = 0
  for i in eachindex(xi)
    if !ismissing(xi[i])
      ind_nna[i] = i
      num_nna = num_nna + 1
    else
      ind_nna[i] = missing
    end
  end
  quadgk(x -> pGi_fun(x, Sigma, p, ind_nna, num_nna), 0, 1)[1]
end

pGi_fun = function (prob, Sigma, p, ind_nna, num_nna)
  X = Matrix{Float64}(undef, length(prob), num_nna)
  for i in eachindex(prob)
    X[i, :] = xi[ind_nna]
  end
  X2 = sign.(X) .* exp.(log.(abs.(X)) .- log(qF(prob, p, D)))
  gauss_res = Vector{Float64}(undef, size(X2)[1])
  for i in 1:eachindex(X2[:, 1])
    gauss_res[i] = mvnormcdf(MvNormal(Sigma[ind_nna, ind_nna]), repeat([-Inf], length(X2[i, :])), X2[i, :])[1]
  end
  gauss_res
end
##

# Multivariate density function (PDF)
dG = function (x, Sigma, p)
  if isa(x, Vector)
    (n, D) = (size(x)[1], 1)
    x = reshape(x, n, D)
  elseif isa(x, Matrix)
    (n, D) = size(x)
  else
    x = reshape([x], 1, 1)
    (n, D) = (1, 1)
  end
  val = Vector{Float64}(undef, n)
  for i in 1:n
    val[i] = dGi(x[i, :], Sigma, p, D)
  end
  return val
end

dGi = function (xi, Sigma, p, D)
  ind_nna = Vector{Int64}(undef, length(xi)) # this should probably be rewritten without loop
  num_nna = 0
  for i in eachindex(xi)
    if !ismissing(xi[i])
      ind_nna[i] = i
      num_nna = num_nna + 1
    else
      ind_nna[i] = missing
    end
  end
  val = quadgk(x -> dGi_fun(x, p, Sigma, D, ind_nna, num_nna), 0, 1)[1]
  return val
end

dGi_fun = function (prob, p, Sigma, D, ind_nna, num_nna)
  X = Matrix{Float64}(undef, length(p), num_nna)
  for i in eachindex(p)
    X[i, :] = xi[ind_nna]
  end
  log_qF = log( qF(prob, p, D) )
  X2 = sign.(X) .* exp.(log.(abs.(X)) .- log_qF)
  gauss_res = Vector{Float64}(undef, size(X2)[1])
  for i in 1:eachindex(X2[:, 1])
    gauss_res[i] = exp( pdf( MvLogNormal(Sigma[ind_nna, ind_nna]), X2[i, :] ) .- num_nna * log_qF)
  end
  return gauss_res
end
##

# Partial derivatives of the distribution function
# leaving this for now 2022-12-14 09:22
# pGI = function (x, I, Sigma, p)
  
# end

# dGIi = function (xi, I)
  
# end

# Random generator from the joint distribution G
rG = function (n, d, Sigma, p)
  R = rF(n, p, d)
  W = rand(MvNormal(Sigma), n)
  R .* W
end

###################################################################################################################################
### Copula, Partial derivatives, Copula Density Function, and Copula random generator for the Gaussian scale mixture model X=RW ###
###################################################################################################################################

# Copula distribution (CDF)
pC = function (u, Sigma, p) # u should be a vector of U(0,1)
  pG( qG1(u, p), Sigma, p)
end

# Copula density (PDF)
dC = function (u, Sigma, p)
  if isa(u, Vector)
     u = reshape(u, length(u), 1)
  elseif isa(u, Matrix)
    u = u
  else
    u = reshape([u], 1, 1)
  end
  qG1_val = qG1(u, p)
  dG(qG1_val, Sigma, p) - sum(dG1(qG1_val, p), dims = 2)
end


# Partial derivatives of the copula distribution C
# leaving this for the now 2022-12-14
# dCI = function ()
  
# end

# Random generator from the copula
rC = function (n, d, Sigma, p)
  pG1(rG(n, d, Sigma, p), p)
end