## R Script for simulating data for a simulation study.
## This code corresponds to the Question 1 of the Problem Set 5.
## 
## Updated: February 17, 2025

# Library: --------------------------------------------------------------------
suppressPackageStartupMessages({
  library( tidyverse )
  library( dirmult )
  library( foreach )
  library( doParallel )
})

# Function for simulating data based on Shi (2023): ---------------------------
data_sim_shi <- function( n1, n2, J, pi_gamma, z_case, aPhi = 1, bPhi = 1,
                          aLambda = 1, bLambda = 1, zsum ){
  
  ### (a) Get the "marginal" probability
  pPhi <- rbeta(J/2, aPhi, bPhi)
  pLambda <- rbeta(J/2, aLambda, bLambda)
  
  ### (b) Shi's adjustment
  pPhi_A <- (1 - (z_case/5)) * pPhi
  pLambda_A <- ((sum(pLambda) + ((z_case/5) * sum(pPhi)))/sum(pLambda)) * pLambda
  pPhi_B <- (1 + (z_case/5)) * pPhi
  pLambda_B <- ((sum(pLambda) - ((z_case/5) * sum(pPhi)))/sum(pLambda)) * pLambda
  
  ### (c) At-risk indicator
  gamma_mat <- matrix(rbinom( (n1 + n2) * J, 1, pi_gamma), nrow = n1 + n2, ncol = J)
  
  ### (d) Calculate the probability matrix for each observations and normalize
  prob_mat <- rbind(t(matrix(rep(c(pPhi_A, pLambda_A), n1), ncol = n1)),
                    t(matrix(rep(c(pPhi_B, pLambda_B), n2), ncol = n2)))
  pg <- prob_mat * gamma_mat
  
  ### (f) Generate the Dirichlet random variables
  dat <- matrix(NA, ncol = J, nrow = n1 + n2)
  for(i in 1:(n1 + n2)){
    prob <- rdirichlet(1, 200 * pg[i, ]/sum(pg[i, ]))
    dat[i, ] <- rmultinom(1, zsum, prob)
  }
  
  list( dat = dat, gamma_mat = gamma_mat, prob_mat = prob_mat )
  
}

clus_shi <- function( n, Jnoise, Jsignal, sumNoise, sumSignal, 
                      prop_nonZero, signalDifficult,
                      patMat = NULL ){
  
  ### J = Jnoise + Jsignal is the number of the taxa
  ### n is a vector of the cluster size. ( length of n >= 2 )
  ### Pattern Matrix ( if null we will randomize the order of Taxa )
  
  K <- length( n )
  tt <- ceiling( K/2 )
  n1_index <- 1
  n2_index <- 2
  ListSignal <- NULL
  datSignal <- NULL
  
  if( is.null( patMat ) ){
    
    patMat <- matrix( NA, nrow = K, ncol = Jnoise + Jsignal )
    for( k in 1:K ){
      
      patMat[ k, ] <- sample( 1:( Jnoise + Jsignal ) )
      
    }
    
  }
  
  patMat_ind <- patMat[ rep( 1:K, times = n ), ]
  
  otuTab <- matrix( NA, ncol = Jnoise + Jsignal, nrow = sum( n ) )
  
  ### Generate Noise
  gammaNoise <- matrix( rbinom( sum( n ) * Jnoise, 1, prop_nonZero ), 
                        nrow = sum( n ), ncol = Jnoise )
  datNoise <- matrix( 0, ncol = Jnoise, nrow = sum( n ) )
  
  for( i in 1:sum( n ) ){
    probNoise <- rdirichlet( 1, gammaNoise[i, ] )
    datNoise[i, ] <- rmultinom( 1, sumNoise, probNoise ) 
  }
  
  ### Generate Signal
  for( t in 1:tt ){
    
    n1 <- n[ n1_index ]
    n2 <- tryCatch( n[ n2_index ], error = function( e ){ NULL } )
    if( is.na( n2 ) ){
      n2 <- n[ n1_index ]
    }
    
    ListSignal <- tryCatch( data_sim_shi( n1 = n1, n2 = n2, 
                                          J = Jsignal, pi_gamma = prop_nonZero, 
                                          z_case = signalDifficult,
                                          zsum = sumSignal ),
                            error = function( e ){ NULL } )
    
    while( is.null( ListSignal ) ){
      ListSignal <- tryCatch( data_sim_shi( n1 = n1, n2 = n2, 
                                            J = Jsignal, pi_gamma = propZero, 
                                            z_case = signalDifficult,
                                            zsum = sumSignal ),
                              error = function( e ){ NULL } )
    }
    
    n1_index <- n1_index + 2
    n2_index <- n2_index + 2
    
    datSignal <- rbind( datSignal, ListSignal$dat )
    
  }
  
  ### Combine noise and signal: OTU table
  for( i in 1:sum( n ) ){
    
    raw_dat <- c( datNoise[ i, ], datSignal[ i, ] )
    otuTab[ i, ] <- raw_dat[ patMat_ind[ i, ] ]
    
  }
  
  otuTab
  
}

# 79: -------------------------------------------------------------------------