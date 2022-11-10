library(mvtnorm)
# generalized gamma distribution:
# https://en.wikipedia.org/wiki/Generalized_gamma_distribution
# a = sqrt(8), p = 2, d = n+1
dF <- function(r, n, log = FALSE){
  K = 2^((3 * n + 1) / 2) * gamma((n + 1) / 2)
  if(log){
    return(n * log(r) - r^2/8 - log(K))
  } else {
    return(r^n * exp(-r^2/8) / K)
  }
}

pF = function(r, n, log = FALSE){
  a = sqrt(8)
  p = 2
  d = n + 1
  if(log){
    return(log(pgamma(r^2, d/p, (1/a)^2)))
  } else {
    return(pgamma(r^2, d/p, (1/a)^2))
  }
}

qF = function(p, n, log = FALSE){
  alpha = (n + 1) / 2
  if(log){
    return(0.5 * log(8) + log(qgamma(p, alpha, 1)) / 2)
  } else {
    return(sqrt(8) * qgamma(p, alpha, 1)^(0.5))
  }
}

rF <- function(n, d){
  return( qF(runif(n), d) )
}

#####################################
## Univariate mixture distriubtion ##
#####################################

pG1 <- function(x, log = FALSE){
  xmat <- x
  if(!is.matrix(x)){
    xmat <- matrix(x, nrow = 1)
  }
  n <- nrow(xmat)
  D <- ncol(xmat)
  fun <- function(p, x, n){
    return( pnorm(sign(x) * exp(log(abs(x)) - qF(p, n, TRUE))) )
  }
  val <- matrix(nrow = n, ncol = D)
  for (i in 1:n){
    for(j in 1:D){
      xi <- xmat[i, j]
      if(!is.na(xi)){
        val[i, j] <- integrate(fun, lower = 0, upper = 1, x = xi, n = n, rel.tol = 10^(-3), stop.on.error = FALSE)$value
      }
    }
  }
  if(!is.matrix(x)){
    val <- as.vector(val)
  }
  if(log){
    return( log(val) )	
  } else{
    return( val )
  }
}

qG1 <- function(p, log = FALSE){ ### if p is a vector, output is a vector; if p is a matrix, output is a matrix.
  pmat <- p
  if(!is.matrix(p)){
    pmat <- matrix(p, nrow = 1)
  }
  n <- nrow(pmat)
  D <- ncol(pmat)
  fun <- function(x, p){
    return( pG1(x) - p )
  }
  val <- matrix(nrow = n, ncol = D)
  for (i in 1:n){
    for(j in 1:D){
      pi <- pmat[i, j]
      if(!is.na(pi)){
        if(pi <= 0){
          val[i, j] <- -Inf
        } else if(pi >= 1){
          val[i, j] <- Inf
        } else{
          val[i, j] <- uniroot(fun, interval = c(-10^2, 10^2), p = pi, extendInt = 'yes')$root
        }
      }
    }
  }
  if(!is.matrix(p)){
    val <- as.vector(val)
  }
  if(log){
    return( sign(val) * log(abs(val)) )	
  } else{
    return( val )
  }
}

#Marginal density function (PDF)
dG1 <- function(x, log = FALSE){ ### if x is a vector, output is a vector; if x is a matrix, output is a matrix.
  xmat <- x
  if(!is.matrix(x)){
    xmat <- matrix(x, nrow = 1)
  } 
  n <- nrow(xmat)
  D <- ncol(xmat)
  fun <- function(p, x, n){
    log_qF <- qF(p, n, TRUE)
    return( exp(dnorm(sign(x) * exp(log(abs(x)) - log_qF), log = TRUE) - log_qF) )
  }
  val <- matrix(nrow = n, ncol = D)
  for (i in 1:n){
    for(j in 1:D){
      xi <- xmat[i, j]
      if(!is.na(xi)){
        val[i, j] <- integrate(fun, lower = 0, upper = 1, x = xi, n = n, rel.tol = 10^(-3), stop.on.error = FALSE)$value
      }
    }
  }
  if(!is.matrix(x)){
    val <- as.vector(val)
  }
  if(log){
    return( log(val) )	
  } else{
    return( val )
  }
}

#Random generator from marginal distribution G1
rG1 <- function(n, d){
  R <- rF(n, d)
  W <- rnorm(n)
  X <- R * W
  return(X)
}


#######################################
## Multivariate mixture distriubtion ##
#######################################

pG <- function(x, Sigma, d, log = FALSE){ ### x is an nxD matrix; if x is a vector, it is interpreted as a single D-variate vector (not D independent univariate random variables)
  if(!is.matrix(x)){
    x <- matrix(x, nrow = 1)
  }
  pGi <- function(xi){
    ind_nna <- !is.na(xi)
    fun <- function(p, d){
      X <- matrix(xi[ind_nna], ncol = sum(ind_nna), nrow = length(p), byrow = TRUE)
      return( apply(matrix(sign(X) * exp(log(abs(X)) - qF(p, d, TRUE)), ncol = sum(ind_nna)), 1, function(x) mvtnorm::pmvnorm(upper = x, sigma = Sigma[ind_nna, ind_nna])[1]) )
    }
    val <- integrate(fun, lower = 0, upper = 1, d, rel.tol = 10^(-3), stop.on.error = FALSE)$value
    return(val)
  }
  val <- apply(x, 1, pGi)
  if(log){
    return( log(val) )	
  } else{
    return( val )
  }
}

dG <- function(x, Sigma, d, log = FALSE){
  if(!is.matrix(x)){
    x <- matrix(x, nrow = 1)
  }
  dGi <- function(xi){
    ind_nna <- !is.na(xi)
    fun <- function(p, d){
      X <- matrix(xi[ind_nna], ncol = sum(ind_nna), nrow = length(p), byrow = TRUE)
      log_qF <- qF(p, d, TRUE)
      return(exp(mvtnorm::dmvnorm(sign(X) * exp(log(abs(X)) - log_qF), sigma = Sigma[ind_nna, ind_nna], log = TRUE) - sum(ind_nna) * log_qF))
    }
    val <- integrate(fun, lower = 0, upper = 1, d = d, rel.tol = 10^(-3), stop.on.error = FALSE)$value
    return(val)
  }
  val <- apply(x, 1, dGi)
  if(log){
    return( log(val) )	
  } else{
    return( val )
  }
}

dGI <- function(x, I, Sigma, log = FALSE){
  if(!is.list(x)){
    if(!is.matrix(x)){
      x <- matrix(x, nrow = 1)
    }
    x <- as.list(data.frame(t(x)))
  }
  D <- length(x[[1]])
  n <- length(x)
  #I is the index vector for partial derivatives
  if(!is.list(I)){
    I <- matrix(I, n, length(I), byrow = TRUE)
    I <- as.list(data.frame(t(I)))
  }
  
  dGIi <- function(xi, I){
    ind_nna <- !is.na(xi)
    nI <- length(I)
    #parameters for the conditional distribution of x[I^c] | x[I]
    Sigma_II <- matrix(Sigma[I, I], ncol = nI) #I need to force a matrix in case nI=1
    Sigma_II_m1 <- solve(Sigma_II)
    Sigma_IcI <- matrix(Sigma[-c(I, which(!ind_nna)), I], ncol = nI) #same problem here
    Sigma_IcIc <- matrix(Sigma[-c(I, which(!ind_nna)), -c(I, which(!ind_nna))], ncol = sum(ind_nna) - nI)
    Sigma_IIc <- t(Sigma_IcI)
    Mu1 <- c(Sigma_IcI %*% Sigma_II_m1 %*% xi[I])
    Sig1 <- Sigma_IcIc - Sigma_IcI %*% Sigma_II_m1 %*% Sigma_IIc
    #function of r to be integrated (needs to be defined for different values of r (= r is a vector))
    fun <- function(p, n){
      X <- matrix(xi, ncol = D, nrow = length(p), byrow = TRUE)
      log_qF <- qF(p, n, TRUE)
      XI_centered <- X[ , -c(I, which(!ind_nna))] - matrix(Mu1, ncol = sum(ind_nna) - nI, nrow = length(p), byrow = TRUE)
      val <- mvtnorm::dmvnorm(matrix(sign(X[ , I]) * exp(log(abs(X[ , I])) - log_qF), ncol = nI), sigma = Sigma_II, log = TRUE) - nI * log_qF
      if(nI < sum(ind_nna)){
        val <- val + apply(matrix(sign(XI_centered) * exp(log(abs(XI_centered)) - log_qF), ncol = sum(ind_nna) - nI), 1, function(x) log(pmin(1, pmax(0, mvtnorm::pmvnorm(upper = x, sigma = Sig1)[1]))))
      }
      val <- exp(val)
      return( val )
    }
    val <- integrate(fun, lower = 0, upper = 1, n, rel.tol = 10^(-3), stop.on.error = FALSE)$value
    return(val)
  }
  val <- mapply(dGIi, xi = x, I = I)
  
  if(log){
    return( log(val) )	
  } else{
    return( val )
  }
}

#Random generator from the joint distribution G
rG <- function(n, Sigma, d){
  R <- rF(n, d)
  W <- mvtnorm::rmvnorm(n, sigma = Sigma)
  X <- R * W
  return(X)
}

###################################################################################################################################
### Copula, Partial derivatives, Copula Density Function, and Copula random generator for the Gaussian scale mixture model X=RW ###
###################################################################################################################################

#Copula distribution (CDF)
pC <- function(u, Sigma, d, log = FALSE, RWscale = FALSE){
  if(RWscale){ ### When u is already on the scale of the process RW
    return( pG(u, Sigma, d, log) ) 
  } else{
    return( pG(qG1(u), Sigma, d, log) )
  }
}

#Copula density (PDF)
dC <- function(u, Sigma, log = FALSE, RWscale = FALSE){
  d = nrow(Sigma)
  if(!is.matrix(u)){
    u <- matrix(u, nrow = 1)
  }
  if(RWscale){ ### When u is already on the scale of the process RW
    val <- dG(u, Sigma, d, log) - rowSums(dG1(u, log = TRUE), na.rm = TRUE)
  } else{
    val <- dG(qG1(u), Sigma, d, log = TRUE) - rowSums(dG1(qG1(u), log = TRUE), na.rm = TRUE)
  }
  if(!log){
    val <- exp(val)
  }
  return( val )
}

#Partial derivatives of the copula distribution C
dCI <- function(u, I, Sigma, log = FALSE, RWscale = FALSE){
  d = nrow(Sigma)
  if(!is.list(u)){
    if(!is.matrix(u)){
      u <- matrix(u, nrow = 1)
    }
    u <- as.list(data.frame(t(u)))
  } 
  n <- length(u)
  #I is the index vector for partial derivatives
  if(!is.list(I)){
    I <- matrix(I, n, length(I), byrow = TRUE)
    I <- as.list(data.frame(t(I)))
  }
  fun1 <- function(u, I){
    return( sum(dG1(u[I], log = TRUE)) )
  }
  fun2 <- function(u, I){
    return( sum(dG1(qG1(u[I]), log = TRUE)) )
  }
  if(RWscale){ ### When u is already on the scale of the process RW
    val <- dGI(u, I, Sigma, log = TRUE) - mapply(fun1, u = u, I = I)
  } else{
    val <- dGI(lapply(u, qG1, n = d), I, Sigma, log = TRUE) - mapply(fun2, u = u, I = I)
  }
  if(!log){
    val <- exp(val)
  }
  return( val )
}

#Random generator from the copula
rC <- function(n, Sigma, empirMar = FALSE){
  d = nrow(Sigma)
  X <- rG(n, Sigma, d)
  if(empirMar){
    U <- apply(X, 2, rank) / (n + 1)	
  } else{
    U <- pG1(X)
  }
  return(U)
}