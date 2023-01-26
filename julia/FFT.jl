module MepdCopula

export rC, dC, pC, qF, pF, pG, dG, qG1

using SpecialFunctions, LinearAlgebra, QuadGK, Roots, Distributions, StatsFuns, MvNormalCDF, Random, InvertedIndices
include("./Constants/qFinterval.jl")
using .QFinterval

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
f(x::Real, α::Real) = α/(π*(x-ζ(α))*abs(α-1)) * quadgk(θ -> h(θ, x, α), -θ₀(α), α <= 0.95 ? π/2 : 1.57; atol = 2e-3)[1]
dstable(x::Real, α::Real, γ::Real) = f((x-γ * tan(π*α/2))/γ, α)/γ

dF = function(x::Real, p::Real, d::Int)
  p > 0 && p < 1 || throw(DomainError(p,"must be on (0,1)")) 
  γ = 2^(1-1/p) * cos(π*p/2)^(1/p)
  C = 2^(1+d/2*(1-1/p)) * gamma(1+d/2) / gamma(1+d/(2*p))
  x > 0 ? C*x^(d-3)*dstable(x^(-2), p, γ) : 0
end

pF = function (x::Real, p::Real, d::Int)
  quadgk(x -> dF(x,p,d), 0, x; atol = 2e-3)[1]
end

qF₁(x::Real, prob::Real, p::Real, d::Integer) = pF(x, p, d) - prob


qF = function(prob::Real, p::Real, d::Integer)
  prob > 0 && prob < 1 || throw(DomainError(prob, "must be on (0,1)"))
  try
    find_zero(x -> qF₁(x, prob, p, d), getInterval(prob, p, d) .+ (0., 1.), xatol=2e-3)
  catch e
    #println("wtf: $prob, $p, $d")
    if isa(e, DomainError) || isa(e, ArgumentError)
      try
        if p > 0.95
          upper = 4
        elseif p <= 0.3
          upper = 10000
        else
          upper = 100
        end
        find_zero(x -> qF₁(x, prob, p, d), (1e-3, upper), xatol = 2e-3)
      catch e
        throw(DomainError((prob, p, d), ": bracketing error with endpoint $upper"))
      end
    end
  end
end

rF = function (n::Integer, p::Real, d::Integer)
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
pG1const = function (x::Matrix{Float64}, p::Real)
  (n, D) = size(x)
  val = Matrix{Float64}(undef, n, D)
  for i in 1:n
    for j in 1:D
      xi = x[i, j]
      if !ismissing(xi)
        val[i, j] = quadgk(x -> pG1_fun(x, xi, p, 1), 0, 1; atol = 2e-3)[1]
      end
    end
  end
  return val
end

pG1_fun = function(prob::Real, x::Real, p::Real, d::Int)
  return StatsFuns.normcdf(0.0, 1.0, sign(x) * exp( log(abs(x)) - log(qF(prob, p, d)) ))
end

pG1(x::Matrix{Float64}, p::Real) = pG1const(x,p)
pG1(x::Vector{Float64}, p::Real) = pG1const(reshape(x, (1,length(x))),p)


##

# Marginal quantile function
qG1const = function(prob::Matrix{Float64}, p::Real)
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
            val[i, j] = find_zero(x -> qG1_fun(x, prob_i, p), (-100, 100))
          end
        end
      end
    end
  end
  return val
end

qG1(prob::Matrix{Float64}, p::Real) = qG1const(prob,p)
qG1(prob::Vector{Float64}, p::Real) = qG1const(reshape(prob, (1,length(prob))),p)

qG1_fun = function(x::Real, prob::Real, p::Real)
  x_mat = reshape([x], 1, 1)
  return pG1(x_mat, p) .- prob
end
##

# Marginal density function (PDF)
dG1 = function(x::Matrix{Float64}, p::Real)
  (n, D) = size(x)
  val = Matrix{Float64}(undef, n, D)
  for i in 1:n
    for j in 1:D
      xi = x[i, j]
      if !ismissing(xi)
        val[i, j] = quadgk(x -> dG1_fun(x, xi, p, D), 1e-6, 1; atol = 2e-3)[1]
      end
    end
  end
  return val
end

dG1_fun = function(prob::Real, x::Real, p::Real, d::Integer)
  log_qF = log( qF(prob, p, d) )
  return exp( StatsFuns.normlogcdf(0.0, 1.0, sign(x) * exp(log(abs(x)) - log_qF) - log_qF) )
end
##

# Random generator from marginal distribution G1
rG1 = function(n::Integer, p::Real, d::Integer)
  R = rF(n, p, d)
  W = rand(Normal(), n)
  return R .* W
end


#################################################################################################################################################
### Multivariate Distribution function, Partial derivatives, Density Function, and Random generator for the Gaussian scale mixture model X=RW ###
#################################################################################################################################################

# Multivariate distribution function (CDF)

pGconst = function(x::Matrix{Float64}, Sigma::Matrix{Float64}, p::Real) ### x is an nxD matrix; if x is a vector, it is interpreted as a single D-variate vector (not D independent univariate random variables)
  (n, D) = size(x)
  val = zeros(Float64,n)
  for i in 1:n
    val[i] = pGi(x[i, :], Sigma, p, D)[1]
  end
  return val
end

pG(x::Matrix{Float64}, Sigma::Matrix{Float64}, p::Real) = pGconst(x, Sigma, p)
pG(x::Vector{Float64}, Sigma::Matrix{Float64}, p::Real) = pGconst(reshape(x, (1, size(Sigma,1))), Sigma, p)

pGi = function(xi::Vector{Float64}, Sigma::Matrix{Float64}, p::Real, D::Integer)
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
  return quadgk(x -> pGi_fun(x, xi, Sigma, p, D, ind_nna), 0, 1; atol = 2e-3)[1]
end

pGi_fun = function(prob::Real, xi::Vector{Float64}, Sigma::Matrix{Float64}, p::Real, D::Integer, ind_nna::Vector{Int64})
  X = sign.(xi[ind_nna]) .* exp.(log.(abs.(xi[ind_nna])) .- log(qF(prob, p, D)))
  if length(X) == 1
    val = StatsFuns.normcdf(0.0, Sigma[1], X[1])
  else
    val = mvnormcdf(MvNormal(Sigma[ind_nna, ind_nna]), repeat([-Inf], length(X)), X)[1]
  end
  return val
end
##

#### Ensure precompilation ####
#pG([1.,1.], [1 0.2; 0.2 1], 0.7);

# Multivariate density function (PDF)
dGconst = function(x::Matrix{Float64}, Sigma::Matrix{Float64}, p::Real)
  (n, D) = size(x)
  val = zeros(Float64, n)
  for i in 1:n
    val[i] = dGi(x[i, :], Sigma, p, D)[1]
  end
  return val
end

dG(x::Matrix{Float64}, Sigma::Matrix{Float64}, p::Real) = dGconst(x, Sigma, p)
dG(x::Vector{Float64}, Sigma::Matrix{Float64}, p::Real) = dGconst(reshape(x, (1,size(Sigma,1))), Sigma, p)

dGi = function(xi::Vector{Float64}, Sigma::Matrix{Float64}, p::Real, D::Integer)
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
  return quadgk(x -> dGi_fun(x, xi, p, Sigma, D, ind_nna, num_nna), 1e-8, 1; atol = 2e-3)[1]
end

dGi_fun = function(prob::Real, xi::Vector{Float64}, p::Real, Sigma::Matrix{Float64}, D::Integer, ind_nna::Vector{Int64}, num_nna::Integer)
  log_qF = log(qF(prob, p, D))
  X = sign.(xi[ind_nna]) .* exp.(log.(abs.(xi[ind_nna])) .- log_qF)
  if length(X) == 1
    val = exp(logpdf(Normal(Sigma[1]), X[1]) .- num_nna * log_qF)
  else
    val = exp(logpdf(MvNormal(Sigma[ind_nna, ind_nna]), X) .- num_nna * log_qF)
  end
  return val
end
##

# Partial derivatives of the distribution function
dGI = function(x::Matrix{Float64}, I::Vector{Vector{Int64}}, Sigma::Matrix{Float64}, p::Real) # x a matrix, I a vector of vectors of indices of threshold exceedances for eahc row of x
  (n, D) = size(x)
  res = Vector{Float64}(undef, n)
  for i in 1:n
    res[i] = dGIi(x[i, :], I[i], D, Sigma, p)
  end
  return res
end

dGIi = function(xi::Vector{Float64}, I::Vector{Int64}, D::Integer, Sigma::Matrix{Float64}, p::Real)
  nI = length(I)
  ind_nna = Vector{Int64}(undef, length(xi))
  num_nna = 0
  for i in eachindex(xi) # this should probably be rewritten without loop
    if !ismissing(xi[i])
      ind_nna[i] = i
      num_nna = num_nna + 1
    else
      ind_nna[i] = missing # look up how missing obs will be treated in empirical data
    end
  end
  # parameters for the conditional distribution of x[I^c] | x[I]
  Sigma_II = Sigma[I, I] # might need to reshape to force it to be a matrix
  Sigma_II_m1 = inv(Sigma_II)
  Sigma_IcI = Sigma[Not(I), I] # NEED FIX: should also exclude rows with missing values
  Sigma_IcIc = Sigma[Not(I), Not(I)] # NEED FIX: should also exclude rows and columns with missing values
  Sigma_IIc = permutedims(Sigma_IcI)
  mu_1 = Sigma_IcI * Sigma_II_m1 * xi[I]
  sig_1 = Sigma_IcIc .- Sigma_IcI * Sigma_II_m1 * Sigma_IIc
  return quadgk(x -> dGIi_fun(x, xi, I, p, D, num_nna, nI, Sigma_II, mu_1, sig_1), 0, 1, atol = 2e-3)[1]
end

dGIi_fun = function(prob::Real, xi::Vector{Float64}, I::Vector{Int64}, p::Real, D::Integer, num_nna::Integer, nI::Integer, Sigma_II::Matrix{Float64}, mu_1::Vector{Float64}, sig_1::Matrix{Float64})
  XI_centerd = xi[Not(I)] .- mu_1 # NEED FIX: should also exclude missing values from xi
  log_qF = qF(prob, p, D)
  X = reshape(sign.(xi[I]) .* exp.(log.(abs.(xi[I])) .- log_qF), nI)
  if length(X) == 1
    val = pdf(LogNormal(Sigma_II[1]), X[1]) .- nI .* log_qF
  else
    val = pdf(MvLogNormal(Sigma_II), X) .- nI .* log_qF
  end
  if (nI < num_nna)
    X2 = reshape(sign.(XI_centerd) .* exp.(log.(abs.(XI_centerd)) .- log_qF), length(XI_centerd))
    val = val .+ gauss_fun(X2, sig_1)
  end
  return exp.(val)
end

gauss_fun = function(up::Vector{Float64}, Sigma::Matrix{Float64})
  if length(up) == 1
    gauss_res = map(max, 0, StatsFuns.normcdf(0.0, Sigma[1], up[1]))
  else
    if !isposdef(Sigma)
      sig_chol = cholesky(Positive, Sigma)
      Sigma = PDMat(sig_chol.L * transpose(sig_chol.L))
    end
    gauss_res = map(max, 0, mvnormcdf(MvNormal(Sigma), repeat([-Inf], length(up)), up)[1])
  end
  return log.(map(min, 1, gauss_res))
end
##

# Random generator from the joint distribution G
rG = function(n::Integer, d::Integer, Sigma::Matrix{Float64}, p::Real)
  R = rF(n, p, d)
  W = rand(MvNormal(Sigma), n)
  return R .* permutedims(W)
end

###################################################################################################################################
### Copula, Partial derivatives, Copula Density Function, and Copula random generator for the Gaussian scale mixture model X=RW ###
###################################################################################################################################

# Copula distribution (CDF)
pCconst = function(u::Matrix{Float64}, Sigma::Matrix{Float64}, p::Real) # u should be a vector of U(0,1)
  return pG(qG1(u, p), Sigma, p)
end

pC(u::Matrix{Float64}, Sigma::Matrix{Float64}, p::Real) = pCconst(u,Sigma,p)
pC(u::Vector{Float64}, Sigma::Matrix{Float64}, p::Real) = pCconst(reshape(u, (1,size(Sigma,1))),Sigma,p)

# Copula density (PDF)
dCconst = function(u::Matrix{Float64}, Sigma::Matrix{Float64}, p::Real)
  qG1_val = qG1(u, p)
  return dG(qG1_val, Sigma, p) .- sum(dG1(qG1_val, p), dims = 2)
end

dC(u::Matrix{Float64}, Sigma::Matrix{Float64}, p::Real) = dCconst(u,Sigma,p)
dC(u::Vector{Float64}, Sigma::Matrix{Float64}, p::Real) = dCconst(reshape(u, (1,size(Sigma,1))),Sigma,p)

# Partial derivatives of the copula distribution C
dCI = function(u::Matrix{Float64}, I::Vector{Vector{Int64}}, Sigma::Matrix{Float64}, p::Real)
  (n, D) = size(u)
  res_1 = Matrix{Float64}(undef, n, D)
  res_2 = Vector{Float64}(undef, n)
  for i in 1:n
    res_1[i, :] = qG1(reshape(u[i, :], 1, D), p)
    res_2[i] = dCI_fun(u[i, :], I[i], p)
  end
  return dGI(res_1, I, Sigma, p) .- res_2
end

dCI_fun = function(u_i::Vector{Float64}, I_i::Vector{Int64}, p::Real)
  u_i_mat = reshape(u_i[I_i], 1, length(I_i))
  return sum(dG1(qG1(u_i_mat, p), p))
end

# Random generator from the copula
rC = function(n::Real, d::Real, Sigma::Matrix{Float64}, p::Real)
  return pG1(rG(n, d, Sigma, p), p)
end

# OBS: ensures  precompilation!
dG([.23, .8], [1 0.2; 0.2 1], 0.7);
pG([.23, .8], [1 0.2; 0.2 1], 0.7);

end