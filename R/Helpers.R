### Code: Simulate only the signal taxa
sim_Signal <- function(N, J, pi_gamma, z_case, aPhi, bPhi, aLambda, bLambda, zsum){
  
  ### (a) Get the "marginal" probability
  J1 <- round(J/2)
  pPhi <- rbeta(J1, aPhi, bPhi)
  pLambda <- rbeta(J - J1, aLambda, bLambda)
  
  ### (b) Shi's adjustment
  pPhi_A <- (1 - (z_case/5)) * pPhi
  pLambda_A <- ((sum(pLambda) + ((z_case/5) * sum(pPhi)))/sum(pLambda)) * pLambda
  pPhi_B <- (1 + (z_case/5)) * pPhi
  pLambda_B <- ((sum(pLambda) - ((z_case/5) * sum(pPhi)))/sum(pLambda)) * pLambda
  
  ### (c) At-risk indicator
  gamma_mat <- matrix(rbinom(N * J, 1, pi_gamma), nrow = N, ncol = J)
  
  ### (d) Calculate the probability matrix for each observations and normalize
  
  N1 <- round(N/2)
  N2 <- N - N1
  
  prob_mat <- rbind(t(matrix(rep(c(pPhi_A, pLambda_A), N1), ncol = N1)),
                    t(matrix(rep(c(pPhi_B, pLambda_B), N2), ncol = N2)))
  pg <- prob_mat * gamma_mat
  
  ### (f) Generate the Dirichlet random variables
  dat <- matrix(NA, ncol = J, nrow = N)
  for(i in 1:N){
    prob <- rdirichlet(1, 200 * pg[i, ]/sum(pg[i, ]))
    dat[i, ] <- rmultinom(1, zsum, prob)
  }
  
  list(dat = dat, gamma_mat = gamma_mat, prob_mat = prob_mat)
  
}

### Code: Simulate the data
sim_clusDat <- function(N, Jnoise, Jsignal, pZero, ZSumNoise, ZSumSignal, 
                        seed, shuffle = TRUE, caseSignal = 3, aPhi = 1, bPhi = 1, 
                        aLambda = 1, bLambda = 1){
  
  set.seed(seed)
  
  ### Check the required packages
  if(!require(dirmult)){
    stop("Missing Package: dirmult")
  }
  
  pi_gamma <- 1 - pZero
  
  ### Generate the noise
  gammaNoise <- matrix(rbinom(N * Jnoise, 1, pi_gamma), 
                       nrow = N, ncol = Jnoise)
  datNoise <- matrix(0, ncol = Jnoise, nrow = N)
  for(i in 1:N){
    probNoise <- rdirichlet(1, gammaNoise[i, ])
    datNoise[i, ] <- rmultinom(1, ZSumNoise, probNoise)
  }
  
  ### Generate the signal
  ListSignal <- sim_Signal(N, Jsignal, pi_gamma, caseSignal, 
                           aPhi, bPhi, aLambda, bLambda, ZSumSignal)
  
  ### Cluster Assignment
  simulated_ci <- c(rep(1, round(N/2)), rep(2, N - round(N/2)))
  simulated_dat <- cbind(datNoise, ListSignal$dat)
  
  ### Create the matrix for storing the final result
  if(shuffle == TRUE){
    index <- sample(1:N)
    simulated_ci <- simulated_ci[index]
    simulated_dat <- simulated_dat[index, ]
  }
  
  list(dat = simulated_dat, c = simulated_ci)
  
}

### Code: Implement the model
ZIDM_DSDM3 <- function(dat, iter, Kmax, nxi_split, theta, s2, s2MH, MHadapt,
                       thin, seed){
  
  n <- nrow(dat)
  J <- ncol(dat)
  
  ### Check the specification: Kmax
  if(Kmax > n){
    Kmax <- n
    print("Adjusted: Kmax = n.")
  }
  
  if(Kmax == 1){
    Kmax <- 2
    print("Adjusted: Kmax = 2 as Kmax should be greater than 1.")
  }
  
  ### Check the specification: nxi_split
  if(nxi_split < 1){
    stop("The number of changed xi (nxi_split) must be non-zero integer.")
  }
  
  if(nxi_split > J){
    nxi_split <- J
    print("Adjusted: nxi_split = J.")
  }
  
  ### Check the specification: MHadapt
  if(MHadapt < 1){
    print("Use the Adaptive MH since the beginning.")
  }
  
  if(MHadapt > iter){
    print("Not Using the Adaptive MH for the whole process")
  }
  
  ### Check the specification: thin
  if(thin < 1){
    stop("The number of thinning (thin) must be non-zero integer.")
  }
  
  set.seed(seed)
  mod_adaptive(iter = iter, Kmax = Kmax, nbeta_split = nxi_split, 
               z = as.matrix(dat),
               atrisk_init = matrix(1, nrow = n, ncol = J),
               beta_init = matrix(0, nrow = Kmax, ncol = J),
               ci_init = rep(0, n), theta = theta, mu = 0, s2 = s2,
               s2_MH = s2MH, t_thres = MHadapt, launch_iter = 30,
               r0g = 1, r1g = 1, r0c = 1, r1c = 1, thin = thin)
  
}

### Code: Number of active cluster via MCMC
uniqueCLUS <- function(resultList){
  apply(resultList$ci_result, 1, function(x){length(unique(x))})
}

### Code: Final Cluster Assignment
finalCLUS <- function(resultList, burn_in, seed){
  
  ### Check the required packages
  if(!require(salso)){
    stop("Missing Package: salso")
  }
  
  suppressWarnings(as.numeric(salso(resultList$ci_result[-(1:burn_in), ])))
  
}

### Code: Evaluate the cluster performance (ARI and Jaccard)
ariCLUS <- function(c1, c2){
  
  ### Check the required packages
  if(!require(mclustcomp)){
    stop("Missing Package: mclustcomp")
  }
  
  suppressWarnings(mclustcomp(c1, c2)[1, 2])
  
}





