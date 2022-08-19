library(libstableR)

## mixture density
dF = function(x, p, d = 1, log = FALSE){
  if(p >= 1 || p <= 0){
    stop("p must be (0,1)")
  }
  n = length(x)
  val = numeric(length(x))
  for(i in 1:n){
    if(x[i] > 0){
      gam = 2^(1-1/p)*(cos(pi*p/2))^(1/p)
      C = 2^(1+d/2*(1-1/p)) * gamma(1+d/2) / gamma(1+d/(2*p))
      val = C*x^(d-3)*stable_pdf(x^(-2), c(p, 1, gam, 0), 1)
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
pF = function(x, p, d = 1, log = FALSE){
  return((integrate(dF, 0, x, p = p, d = d)$value))
}

## mixture quantile function
qF = function(prob, p, d = 1, log = FALSE){
  fun = function(x, prob, p, d = 1) pF(x, p, d) - prob
  return(uniroot(fun, prob = prob, p = p, d = d, lower = 0, upper=100)$root)
}

# tar rätt lång tid att simulera pga uniroot
rF = function(n, p, d = 1){
  val = numeric(n)
  for(i in 1:n){
    val[i] = qF(runif(1), p, d)
  }
  return(val)
}

