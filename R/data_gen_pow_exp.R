corr_fun <- function(h, param){
  return( exp( -(h / exp(param[1]) )^param[2]) )
}


# General shape parameter
pow_exp_data_gen <- function(n_obs, n_sites, param, empirMar){
  
  coord <- matrix(runif(2 * n_sites), ncol = 2)
  
  dists <- rbind(as.vector(matrix(coord[, 1], nrow = n_sites, ncol = n_sites) - matrix(coord[, 1], nrow = n_sites, ncol = n_sites, byrow = TRUE)),
                 as.vector(matrix(coord[, 2], nrow = n_sites, ncol = n_sites) - matrix(coord[, 2], nrow = n_sites, ncol = n_sites, byrow = TRUE)))
  
  D <- matrix(sqrt(dists[1,]^2 + dists[2,]^2), nrow = n_sites, ncol = n_sites)
  
  Sigmab <- corr_fun(D, param)
  
  dat_sim <- rC(n = n_obs, Sigma = Sigmab, par = param[3], empirMar)
  
  return(list(data = dat_sim, coords = coord))
}

# p = 0.5
laplace_data_gen <- function(n_obs, n_sites, param, empirMar){
  
  coord <- matrix(runif(2 * n_sites), ncol = 2)
  
  dists <- rbind(as.vector(matrix(coord[, 1], nrow = n_sites, ncol = n_sites) - matrix(coord[, 1], nrow = n_sites, ncol = n_sites, byrow = TRUE)),
                 as.vector(matrix(coord[, 2], nrow = n_sites, ncol = n_sites) - matrix(coord[, 2], nrow = n_sites, ncol = n_sites, byrow = TRUE)))
  
  D <- matrix(sqrt(dists[1,]^2 + dists[2,]^2), nrow = n_sites, ncol = n_sites)
  
  Sigmab <- corr_fun(D, param)
  
  dat_sim <- rC(n = n_obs, Sigma = Sigmab, empirMar)
  
  return(list(data = dat_sim, coords = coord))
}


