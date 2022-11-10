#############################################
# uncensored powered exponential with p = 0.5
#############################################
copula_powExp_nocens <- function(dat, coord, init_val, ncores = 1, hessian = FALSE, optim = TRUE, method = "Nelder-Mead"){

  n_sites <- apply(!is.na(dat), 1, sum)
	na_obs <- n_sites <= 1
	dat <- dat[!na_obs, ]
	n_sites <- n_sites[!na_obs]
	
	n_obs <- nrow(dat)
	n_sites <- ncol(dat)
  
  #### model for W
	# correlation functions
	fcor <- function(h, param){
	  return(exp(-(h / param[1])^param[2]))
	}
	#conditions on the parameters of the correlation function (I need this such that the nllik function returns Inf if the conditions are not met)
	cond_cor <- function(param){
	  # return(param[2] > 0 & param[2] < 2 & param[1] > 0)
	  return(param[2] > 0 & param[2] < 2)
	}
	
	### negative censored log-likelihood
	nllik <- function(param){
		#check conditions on parameters
		if (!cond_cor(param)){return(10^10)}
		
	  #calculates the distance matrix
		dists <- rbind(as.vector(matrix(coord[, 1], nrow = n_sites, ncol = n_sites) - matrix(coord[, 1], nrow = n_sites, ncol = n_sites, byrow = TRUE)),
		               as.vector(matrix(coord[, 2], nrow = n_sites, ncol = n_sites) - matrix(coord[, 2], nrow = n_sites, ncol = n_sites, byrow = TRUE)))
		D <- matrix(sqrt(dists[1,]^2 + dists[2,]^2), nrow = n_sites, ncol = n_sites)
		
		#calculate the matrix of correlation in W
		Sigmab <- fcor(D, param)

		tr <- try(chol(Sigmab), TRUE)
    if (is(tr, "try-error")){return(10^10)}
			
		### negative censored log-likelihood for a specific block
		nllik_block <- function(block){ #block is the index of the block (when parallel computing is used)
			if(ncores > 1){
				indmin <- unname(c(0.5, quantile(1:n_obs, c(1:(ncores - 1)) / ncores)))[block]
				indmax <- unname(c(quantile(1:n_obs, c(1:(ncores - 1)) / ncores), n_obs + 0.5))[block]
				ind_block <- c(1:n_obs)[c(1:n_obs) > indmin & c(1:n_obs) <= indmax] ### indices of the specific block
			} else if(ncores == 1){
				ind_block <- c(1:n_obs)
			}
			  
		  #transform data datUc to RW scale using qG
			x <- matrix(qG1(c(dat[ind_block, ])), ncol = ncol(dat[ind_block, ]))
	
			contrib <- sum(dC(x, Sigma = Sigmab, log = TRUE, RWscale = TRUE))
			
			return(-contrib)
		}
	
		nllik_res <- sum(unlist(parallel::parLapply(cl, 1:ncores, nllik_block)))

		return(nllik_res)	
			
	}

	cl <- parallel::makeCluster(ncores, type = "SOCK")
	parallel::clusterExport(cl, ls(envir = .GlobalEnv))
	parallel::clusterEvalQ(cl, library(mvtnorm))

	opt <- optim(init_val, nllik, hessian = hessian, method = "Nelder-Mead") #, control = list(...))

	parallel::stopCluster(cl)

	#results
	mle <- c()
	mle <- opt$par
	z <- list()
	z$mle <- mle
	z$nllik <- opt$val
	z$convergence <- opt$convergence
	z$hessian <- opt$hessian
	if(!is.null(z$hessian)){
		z$std <- sqrt(diag(solve(z$hessian)))
	}
	z$counts <- opt$counts
	
	return(z)		
}

# generate data
fcor <- function(h, param){
  return(exp(-(h / param[1])^param[2]))
}

n_sites <- 10
set.seed(123); coord <- matrix(runif(2 * n_sites), ncol = 2)
param <- c(1.5, 1.5)

dists <- rbind(as.vector(matrix(coord[, 1], nrow = n_sites, ncol = n_sites) - matrix(coord[, 1], nrow = n_sites, ncol = n_sites, byrow = TRUE)),
               as.vector(matrix(coord[, 2], nrow = n_sites, ncol = n_sites) - matrix(coord[, 2], nrow = n_sites, ncol = n_sites, byrow = TRUE)))
D <- matrix(sqrt(dists[1,]^2 + dists[2,]^2), nrow = n_sites, ncol = n_sites)
Sigmab <- fcor(D, param)
set.seed(321)
dat <- rC(n = 10, Sigma = Sigmab, empirMar = FALSE)

# estimate parameters
copula_powExp_nocens(dat = dat, coord = coord, init_val = c(1, 1), ncores = 5, hessian = FALSE, method = "Nelder-Mead")
#







#############################################
# censored powered exponential with p = 0.5
#############################################
# have not tested 2022-11-10
copula_powExp_cens <- function(dat_U, coord, thresh, init_val, ncores = 1, hessian = FALSE, optim = TRUE, method = "Nelder-Mead"){

  ### censor data below the threshold
	dat_U_cens <- dat_U
	dat_U_cens[dat_U < thres] <- thres
  
  n_sites <- apply(!is.na(dat_U), 1, sum)
	na_obs <- n_sites <= 1
	dat_U_cens <- dat_U_cens[!na_obs, ]
	n_sites <- n_sites[!na_obs]
	
	n_obs <- nrow(dat_U_cens)
	n_sites <- ncol(dat_U_cens)
	
	
	Iexc <- apply(dat_U_cens > thres, 1, which)
	I1 <- lapply(Iexc, length) == n_stats # no censoring
	I2 <- lapply(Iexc, length) > 0 & lapply(Iexc, length) < n_stats # partial censoring
	I3 <- lapply(Iexc, length) == 0 # fully censored
	
	Inexc <- unique(apply(dat_U_cens <= thres, 1, which)[I3])
	Inexc_len <- unlist(lapply(Inexc, length))
	same_vec <- function(vec1, vec2){
		if(length(vec1) != length(vec2)){
			return(FALSE)
		} else{
			return(!any(!(vec1 %in% vec2)))
		}
	}
	compute_Inexc_nb_i <- function(i){
	  return( sum(unlist(lapply(apply(dat_U_cens <= thres, 1, which)[I3], same_vec, vec2 = Inexc[[i]]))) )
	}
	
	cl <- makeCluster(ncores, type = "SOCK")
	Inexc_nb <- unlist(parallel::parLapply(cl, 1:length(Inexc), compute_Inexc_nb_i))	
  stopCluster(cl)
	
	dims_c <- sort(unique(n_stats[I3]))
	nb_dims_c <- c()
	for(i in 1:length(dims_c)){
	  nb_dims_c[i] <- sum(n_stats[I3] == dims_c[i])
	}
	nfc <- sum(I3) # number of fully censored obs
	inds <- I1 | I2 # indices for parallel computing of exceedance contributions (non-exceedances are treated separately)...

	#### model for W
	# correlation functions
	fcor <- function(h, param){
	  return(exp(-(h / param[1])^param[2]))
	}
	#conditions on the parameters of the correlation function (I need this such that the nllik function returns Inf if the conditions are not met)
	cond_cor <- function(param){
	  return(param[2] > 0 & param[2] < 2 & param[1] > 0)
	}
	
	### negative censored log-likelihood
	nllik <- function(param){
		#check conditions on parameters
		if (!cond_cor(param)){return(10^10)}
		
	  #calculates the distance matrix
		dists <- rbind(as.vector(matrix(coord[, 1], nrow = n_sites, ncol = n_sites) - matrix(coord[, 1], nrow = n_sites, ncol = n_sites, byrow = TRUE)),
		               as.vector(matrix(coord[, 2], nrow = n_sites, ncol = n_sites) - matrix(coord[, 2], nrow = n_sites, ncol = n_sites, byrow = TRUE)))
		D <- matrix(sqrt(dists[1,]^2 + dists[2,]^2), nrow = n_sites, ncol = n_sites)
		
		#calculate the matrix of correlation in W
		Sigmab <- fcor(D, param)

		tr <- try(chol(Sigmab), TRUE)
    if(is(tr, "try-error")){return(10^10)}
		
    #transform data dat_U_cens to RW scale using qG
		xc <- matrix(nrow = n_obs, ncol = n_stat)
		xc[which(dat_U_cens > thres)] <- unlist(parallel::parLapply(cl, dat_U_cens[which(dat_U_cens > thres)], qG1, par = param))
		xthres <- qG1(thres, par = param)
		xc[which(dat_U_cens == thres)] <- xthres
		
		#fix random seed (and save the current random seed to restore it at the end)
		oldSeed <- get(".Random.seed", mode="numeric", envir=globalenv())
		
			
		### negative censored log-likelihood for a specific block
		nllik_block <- function(block){ #block is the index of the block (when parallel computing is used)
			if(ncores > 1){
				indmin <- unname(c(0.5, quantile(1:sum(inds), c(1:(ncores - 1)) / ncores)))[block]
				indmax <- unname(c(quantile(1:sum(inds), c(1:(ncores - 1)) / ncores), sum(inds) + 0.5))[block]
				ind_block <- c(1:sum(inds))[c(1:sum(inds)) > indmin & c(1:sum(inds)) <= indmax] ### indices of the specific block
			} else if(ncores == 1){
				ind_block <- c(1:sum(inds))
			}
			  
		  set.seed(123)
			#sum the contributions for the different cases
			contrib1 <- contrib2 <- 0
			if (sum(I1[inds][ind_block]) > 0){ #no censoring
				contrib1 <- sum(dC(xc[inds, ][ind_block, ][I1[inds][ind_block], ], Sigma = Sigmab, par = exp(a), log = TRUE, RWscale = TRUE))
			}
			if (sum(I2[inds][ind_block]) > 0){ #partial censoring
				contrib2 <- sum(dCI(as.list(data.frame(t(xc[inds, ][ind_block, ])))[I2[inds][ind_block]], I = Iexc[inds][ind_block][I2[inds][ind_block]], Sigma = Sigmab, par = exp(a), log = TRUE, RWscale = TRUE))
			}	
			return(-(contrib1 + contrib2))
		}
	
		nllik_res <- sum(unlist(parallel::parLapply(cl, 1:ncores, nllik_block)))

		#fully censored obs
		compute_contrib3 <- function(i){
			set.seed(123456)
			return( Inexc_nb[i] * pC(rep(xthres, Inexc_len[i]), Sigma = Sigmab[Inexc[[i]], Inexc[[i]]], par = exp(a), log = TRUE, RWscale = TRUE) )
		}
		contribs3 <- unlist(parallel::parLapply(cl, 1:length(Inexc), compute_contrib3))
		contrib3 <- sum(contribs3)
		
		#restore random seed to its previous value
		assign(".Random.seed", oldSeed, envir = globalenv())
		return(nllik_res - contrib3)
	}

	### optimization with respect to parameters that are not fixed	
	if (optim == TRUE){
		# init_val2 <- init_val[which(!fixed)]
		# nllik2 <- function(par_opt){
		# 	all_par <- init_val
		# 	all_par[which(!fixed)] <- par_opt
		# 	return(nllik(a=all_par[1:2],b=all_par[-c(1:2)]))
		# }

		cl <- makeCluster(ncores, type = "SOCK")
		clusterExport(cl, ls(envir = .GlobalEnv))
		clusterEvalQ(cl, library(mvtnorm))

		# opt <- optim(init_val2,nllik2,hessian=hessian,method=method,control=list(...))
		opt <- optim(init_val, nllik, hessian = hessian, method = method, control = list(...))
	
		stopCluster(cl)
	
		#results
		mle <- c()
		# mle[!fixed] <- opt$par
		mle <- opt$par
	   	# mle[fixed] <- init_val[fixed]
		z <- list()
		z$mle <- mle
		z$nllik <- opt$val
		z$convergence <- opt$convergence
		z$hessian <- opt$hessian
		if(!is.null(z$hessian)){
			z$std <- sqrt(diag(solve(z$hessian)))
		}
		z$counts <- opt$counts
		
		return(z)		
	}
	
	# if (optim == FALSE){
	# 	cl <- makeCluster(ncores, type = "SOCK")
	# 	clusterExport(cl, ls(envir = .GlobalEnv))
	# 	clusterEvalQ(cl, library(mvtnorm))
	# 
	# 	return(nllik(a=init_val[1:2], b=init_val[-c(1:2)]))
	# 	
	# 	stopCluster(cl)
	# }
}

















distance_fun <- function(lambda, nu, coord = coord, n_sites = n_sites){
  
  covarMat = matrix(0, n_sites, n_sites)
  
  for(k in 1:n_sites){
    for(l in 1:n_sites){
      
      if(k == l){
        covarMat[k, l] <- 2 * (sqrt(sum(coord[k, ]^2)) / lambda)^nu
        } else{
          var1 = sqrt(sum(coord[k, ]^2))
          var2 = sqrt(sum(coord[l, ]^2))
          covars = sqrt(sum((coord[k, ] - coord[l, ])^2))
          
          covarMat[k, l] = lambda^(-nu) * (var1^nu + var2^nu - covars^nu)
        }
    }
  }
  return(covarMat)
}

corrMatrixFun = function(data, lambda, nu){

  numOfCat = length(data)
  corrMatrix = matrix(NA, n_sites, n_sites)
  for(i in 1:numOfCat){
    for(j in 1:numOfCat){
      d = abs(data[i] - data[j]) # compute distances
      corr = exp(- (d / lambda)^nu) # exponential similarity
      corrMatrix[i, j] = corr
    }
  }
  return(corrMatrix)
}