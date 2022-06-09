library(libstableR)
library(LaplacesDemon)

# Computes the mixture distribution
h = function(x, p, d){
  gam = 2^(1-1/p)*(cos(pi*p/2))^(1/p)
  C = 2^(1+d/2*(1-1/p)) * gamma(1+d/2) / gamma(1+d/(2*p))
  return(C*x^(d-3)*stable_pdf(x^(-2), c(p, 1, gam, 0), 1)) 
}


f = function(r, x, p, d = 1){
  return(dnorm(x,0,r^2)*h(r,p,d))
}

margf = function(r, x, p, d = 1){
  return(pnorm(x,0,r^2) * h(r,p,d))
}

# Computes second equation of (5) in Extreme scale mixture paper
g = function(p, x, d = 1){
  return((integrate(f, 0, Inf, x = x, p = p, d = d, abs.tol = 10^(-3))$value))
}

# Computes first equation of (5) in Extreme scale mixture paper
G = function(p, x, d = 1){
  return(integrate(margf, 0, Inf, x = x, p = p, d = d, abs.tol = 10^(-3))$value)
}

## Censored stuff, ignore for now ##
fullmarg = function(r, x, p, Sigma){
  pmvnorm(0, x, rep(0, nrow(Sigma)), sigma = Sigma, keepAttr =F) * h(r, p, ncol(Sigma))
}

fullG = function(x, p, Sigma){
  return(integrate(fullmarg, 0, Inf, x = x, p = p, Sigma = Sigma, abs.tol = 10^(-3))$value)
}
####################################


qf = function(x, p, q, d = 1) G(p, x, d) - q
# Computes MEPD quantile
qmvpe = function(q, p, d = 1){
  if(q >= 0.5){
    return(uniroot(qf, p = p, q = q, d = d, lower = 0, upper=10000)$root) 
  } else {
    return(uniroot(qf, p = p, q = q, d = d, lower = -10000, upper=0)$root)
  }
}

# correlation for process covariance
rho = function(s1, s2, lambda, nu = 1){
  return(exp(-(sqrt(c(t(s1 - s2) %*% (s1 - s2)))/lambda)^nu))
}

# builds the covariance matrix based on a K x 2 matrix of spatial positions
buildCovMat = function(grid, lambda){
  n = nrow(grid)
  covMat = matrix(1, n, n)
  for(i in 1:(n-1)){
    for(j in (i+1):n){
      covMat[i,j] = rho(grid[i,], grid[j,], lambda)
      covMat[j,i] = covMat[i,j]
    }
  }
  return(covMat)
}


loglik = function(p, x, Sigma){
  denom = 0
  d = length(x)
  nom = numeric(d)
  for(i in 1:d){
    u = max(c(0.9, x[i]))
    nom[i] = qmvpe(u, p, d)
    denom = denom + log(g(p, nom[i], d))
  }
  return(dmvpe(nom, rep(0, length(x)), Sigma, p, T) - denom)
}

# Computes the full negative log likelihood with no censoring
lik = function(p, U, Sigma) sum(apply(U, 1, function(x){-loglik(p, x, Sigma)}))

# Computes the full negative log likelihood with all observations censored
censLik = function(p, U, Sigma){
  x = numeric(nrow(Sigma))
  for(i in 1:length(x)){
    x[i] = qmvpe(0.999, p, nrow(Sigma))
  }
  return(log(fullG(x, p, Sigma)))
}
