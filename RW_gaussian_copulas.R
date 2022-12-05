library(libstableR)

## mixture density
dF = function(x, par, d = 1, log = FALSE){
  if(par >= 1 || par <= 0){
    stop("par must be (0,1)")
  }
  n = length(x)
  val = numeric(length(x))
  for(i in 1:n){
    if(x[i] > 0){
      gam = 2^(1 - 1 / par) * (cos(pi * par / 2))^(1 / par)
      C = 2^(1 + d / 2 * (1 - 1 / par)) * gamma(1 + d / 2) / gamma(1 + d / (2 * par))
      val = C * x^(d - 3) * libstableR::stable_pdf(x^(-2), c(par, 1, gam, 0), 1)
    } else {
      val[i] = 0
    }
  }
  if(log){
    return(log(val))
  } else {
    return(val)
  }
}

## Mixture CDF
pF = function(x, par, d = 1, log = FALSE){
  return((integrate(f = Vectorize(dF), lower = 0, upper = x, par = par, d = d, rel.tol = 10^(-1))$value))
}

## mixture quantile function
upperPoint = function(p, d){
  if(p >= 0.35 && p < 0.45){
    return(200 - 400*p + 3*d)
  } else if(p >= 0.45 && p < 0.6){
    return(46 - 68*p + 2*d)
  } else if(p >= 0.6 && p < 0.75){
    return(13.3 - 14.5*p + 0.5*d)
  } else if(p>=0.75){
    return(6.3 - 5.3*p + 0.25*d)
  } else {
    return(100)
  }
}

qF = function(prob, par, d = 1, log = FALSE){
  fun = function(x, prob, par, d = 1){
    pF(x, par, d) - prob
  }
  tryCatch(
    {
      return(uniroot(f = Vectorize(fun), interval = c(0, upperPoint(par, d)), prob = prob, 
                     par = par, d = d, maxiter = 50, tol = 1e-2)$root)   
    }, error = function(cond) {
      return(uniroot(f = Vectorize(fun), interval = c(0, 100), prob = prob, par = par, d = d)$root)   
    }
  )
}


# tar tid att simulera pga uniroot
rF = function(n, par, d = 1){
  val = numeric(n)
  for(i in 1:n){
    val[i] = qF(runif(1), par, d)
  }
  return(val)
}


#####################################
## Univariate mixture distriubtion ##
#####################################

#Marginal distribution function (CDF)
pG1 <- function(x, par, log = FALSE){ ### if x is a vector, output is a vector; if x is a matrix, output is a matrix.
	xmat <- x
	if(!is.matrix(x)){
		xmat <- matrix(x, nrow = 1)
	}
	n <- nrow(xmat)
	D <- ncol(xmat)
	fun <- function(p, x, par){
		return( pnorm(sign(x) * exp(log(abs(x)) - qF(p, par, TRUE))) )
	}
	val <- matrix(nrow = n, ncol = D)
	for (i in 1:n){
		for(j in 1:D){
			xi <- xmat[i, j]
			if(!is.na(xi)){
				val[i, j] <- integrate(f = Vectorize(fun), lower = 0, upper = 1, x = xi, par = par, rel.tol = 10^(-1), stop.on.error = FALSE)$value
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

#Marginal quantile function
qG1 <- function(p, par, log = FALSE){ ### if p is a vector, output is a vector; if p is a matrix, output is a matrix.
	pmat <- p
	if(!is.matrix(p)){
		pmat <- matrix(p, nrow = 1)
	}
	n <- nrow(pmat)
	D <- ncol(pmat)
	fun <- function(x, p, par){
		return( pG1(x, par) - p )
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
					val[i, j] <- uniroot(f = Vectorize(fun), interval = c(-10^2, 10^2), p = pi, par = par, extendInt = 'yes')$root
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
dG1 <- function(x, par, log = FALSE){ ### if x is a vector, output is a vector; if x is a matrix, output is a matrix.
	xmat <- x
	if(!is.matrix(x)){
		xmat <- matrix(x, nrow = 1)
	} 
	n <- nrow(xmat)
	D <- ncol(xmat)
	fun <- function(p, x, par){
		log.qF <- qF(p, par, TRUE)
		return( exp(dnorm(sign(x) * exp(log(abs(x)) - log.qF), log = TRUE) - log.qF) )
	}
	val <- matrix(nrow = n, ncol = D)
	for (i in 1:n){
		for(j in 1:D){
			xi <- xmat[i, j]
			if(!is.na(xi)){
				val[i, j] <- integrate(f = Vectorize(fun), lower = 0, upper = 1, x = xi, par = par, rel.tol = 10^(-1), stop.on.error = FALSE)$value
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
rG1 <- function(n, par){
	R <- rF(n, par)
	W <- rnorm(n)
	X <- R * W
	return(X)
}

#################################################################################################################################################
### Multivariate Distribution function, Partial derivatives, Density Function, and Random generator for the Gaussian scale mixture model X=RW ###
#################################################################################################################################################
	
#Multivariate distribution function (CDF)
pG <- function(x, Sigma, par, log = FALSE){ ### x is an nxD matrix; if x is a vector, it is interpreted as a single D-variate vector (not D independent univariate random variables)
	if(!is.matrix(x)){
		x <- matrix(x, nrow = 1)
	}
	pGi <- function(xi){
		ind.nna <- !is.na(xi)
		fun <- function(p, par){
			X <- matrix(xi[ind.nna], ncol = sum(ind.nna), nrow = length(p), byrow = TRUE)
			return( apply(matrix(sign(X) * exp(log(abs(X)) - qF(p, par, TRUE)), ncol = sum(ind.nna)),1 , function(x) mvtnorm::pmvnorm(upper = x, sigma = Sigma[ind.nna,ind.nna])[1]) )
		}
		val <- integrate(f = Vectorize(fun), lower = 0, upper = 1, par = par, rel.tol = 10^(-1), stop.on.error = FALSE)$value
		return(val)
	}
	val <- apply(x, 1, pGi)
	if(log){
		return( log(val) )	
	} else{
		return( val )
	}
}

#Multivariate density function (PDF)
dG <- function(x, Sigma, par, log = FALSE){
	if(!is.matrix(x)){
		x <- matrix(x, nrow = 1)
	}
	dGi <- function(xi){
		ind.nna <- !is.na(xi)
		fun <- function(p, par){
			X <- matrix(xi[ind.nna], ncol = sum(ind.nna), nrow = length(p), byrow = TRUE)
			log.qF <- qF(p, par, TRUE)
			return(exp(mvtnorm::dmvnorm(sign(X) * exp(log(abs(X)) - log.qF), sigma = Sigma[ind.nna, ind.nna], log = TRUE) - sum(ind.nna) * log.qF))
		}
		val <- integrate(f = Vectorize(fun), lower = 0, upper = 1, par = par, rel.tol = 10^(-1), stop.on.error = FALSE)$value
		return(val)
	}
	val <- apply(x, 1, dGi)
	if(log){
		return( log(val) )	
	} else{
		return( val )
	}
	# }
}

#Partial derivatives of the distribution function
dGI <- function(x, I, Sigma, par, log = FALSE){
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
		ind.nna <- !is.na(xi)
		nI <- length(I)
		#parameters for the conditional distribution of x[I^c] | x[I]
		Sigma_II <- matrix(Sigma[I, I], ncol = nI) #I need to force a matrix in case nI=1
		Sigma_II_m1 <- solve(Sigma_II)
		Sigma_IcI <- matrix(Sigma[-c(I, which(!ind.nna)), I], ncol = nI) #same problem here
		Sigma_IcIc <- matrix(Sigma[-c(I, which(!ind.nna)), -c(I, which(!ind.nna))], ncol = sum(ind.nna) - nI)
		Sigma_IIc <- t(Sigma_IcI)
		Mu1 <- c(Sigma_IcI %*% Sigma_II_m1 %*% xi[I])
		Sig1 <- Sigma_IcIc - Sigma_IcI %*% Sigma_II_m1 %*% Sigma_IIc
		#function of r to be integrated (needs to be defined for different values of r (= r is a vector))
		fun <- function(p, par){
			X <- matrix(xi, ncol = D, nrow = length(p), byrow = TRUE)
			log.qF <- qF(p, par, TRUE)
			XI.centered <- X[, -c(I, which(!ind.nna))] - matrix(Mu1, ncol = sum(ind.nna) - nI, nrow = length(p), byrow = TRUE)
			val <- mvtnorm::dmvnorm(matrix(sign(X[, I]) * exp(log(abs(X[, I])) - log.qF), ncol = nI), sigma = Sigma_II, log = TRUE) - nI * log.qF
			if(nI < sum(ind.nna)){
				val <- val + apply(matrix(sign(XI.centered) * exp(log(abs(XI.centered)) - log.qF), ncol = sum(ind.nna) - nI), 1,
				                   function(x) log(pmin(1, pmax(0, mvtnorm::pmvnorm(upper = x, sigma = Sig1)[1]))))
			}
			val <- exp(val)
			return( val )
		}
		val <- integrate(Vectorize(fun), lower = 0, upper = 1, par, rel.tol = 10^(-1), stop.on.error = FALSE)$value
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
rG <- function(n, Sigma, par){
	R <- rF(n, par)
	W <- mvtnorm::rmvnorm(n = n, sigma = Sigma)
	X <- R * W
	return(X)
}

###################################################################################################################################
### Copula, Partial derivatives, Copula Density Function, and Copula random generator for the Gaussian scale mixture model X=RW ###
###################################################################################################################################

#Copula distribution (CDF)
pC <- function(u, Sigma, par, log = FALSE, RWscale = FALSE){
	if(RWscale){ ### When u is already on the scale of the process RW
		return( pG(u, Sigma, par, log) ) 
	} else{
		return( pG(qG1(u, par), Sigma, par, log) )
	}
}

#Copula density (PDF)
dC <- function(u, Sigma, par, log = FALSE, RWscale = FALSE){
	if(!is.matrix(u)){
		u <- matrix(u, nrow = 1)
	}
	if(RWscale){ ### When u is already on the scale of the process RW
		val <- dG(u, Sigma, par, log) - rowSums(dG1(u, par, log = TRUE), na.rm = TRUE)
	} else{
		val <- dG(qG1(u, par), Sigma, par, log = TRUE) - rowSums(dG1(qG1(u, par), par, log = TRUE), na.rm = TRUE)
	}
	if(!log){
		val <- exp(val)
	}
	return( val )
}

#Partial derivatives of the copula distribution C
dCI <- function(u, I, Sigma, par, log = FALSE, RWscale = FALSE){
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
	fun1 <- function(u,I){
		return( sum(dG1(u[I], par, log = TRUE)) )
	}
	fun2 <- function(u, I){
		return( sum(dG1(qG1(u[I], par), par, log = TRUE)) )
	}
	if(RWscale){ ### When u is already on the scale of the process RW
		val <- dGI(u, I, Sigma, par, log = TRUE) - mapply(fun1, u = u, I = I)
	} else{
		val <- dGI(lapply(u, qG1, par = par), I, Sigma, par, log = TRUE) - mapply(fun2, u = u ,I = I)
	}
	if(!log){
		val <- exp(val)
	}
	return( val )
}

#Random generator from the copula
rC <- function(n, Sigma, par, empirMar = FALSE){
	X <- rG(n, Sigma, par)
	if(empirMar){
		U <- apply(X, 2, rank) / (n + 1)	
	} else{
		U <- pG1(x = X, par = par)
	}
	return(U)
}
#