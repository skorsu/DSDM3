## R script for the wrapper function of the main model function.
##
## This file defines a wrapper function for the main function. The wrapper 
## handles model initialization, runs the model, and saves the results.
##
## Updated: May 28, 2025

# Function: Implementing the model --------------------------------------------
DSDM3_ZIDM <- function( otuTab, Km, pi_TB, s2_prior, s2_MH, theta, 
                        a_gamma, b_gamma, Kp_init, s, iter, thin, seed ){
  
  ### Settings
  n <- nrow( otuTab )
  J <- ncol( otuTab )
  A <- s/median( rowSums( otuTab ) )
  
  ### Check the specification: Kmax
  if( Km > n ){
    Km <- n
    print( "Adjusted: Km = n." )
  }
  
  if( Km == 1 ){
    Km <- 2
    print( "Adjusted: Km = 2 as Km should be greater than 1." )
  }
  
  ### Check the specification: thin
  if( thin < 1 ){
    stop( "The number of thinning (thin) must be non-zero integer." )
  }
  
  ### Initialize
  set.seed( seed = seed )
  mean_prior <- log( colMeans( otuTab ) * A )
  mean_prior[ is.infinite( mean_prior ) ] <- log( 0.001 )
  
  cVec <- sample( 1:Kp_init, n, replace = TRUE )
  xiMat <- matrix( 0, ncol = J, nrow = Km )
  for( k in 1:Kp_init ){
    
    l0 <- which( colMeans( otuTab[ which( cVec == k ), ] == 0 ) == 1 )
    if( length( l0 ) == 0 ){
      
      xiMat[ k, ] <- log( colMeans( otuTab[ which( cVec == k ), ] ) * A )
      
    } else {
      
      xiMat[ k, -l0 ] <- log( colMeans( otuTab[ which( cVec == k ), -l0 ] ) * A )
      xiMat[ k, l0 ] <- log( 0.001 )
      
    }
    
  }
  
  lp <- sapply( 1:Km, function( x ){ log_pTB( x, Km = Km, pi = pi_TB ) } )
  
  ### Store the result
  c_store <- matrix( 0, ncol = n, nrow = iter )
  eta_store <- matrix( 0, ncol = Km, nrow = iter )
  Kpv <- rep( -1, iter )
  Kv <- rep( -1, iter )
  xic <- array( 0, dim = c( Km, J, iter/thin ) )
  arc <- array( 0, dim = c( n, J, iter/thin ) )
  s2a_perform <- c( 0 )
  s2a_accept <- c( 0 )
  
  ### Run the model
  set.seed( seed )
  mod_fixed( c_init = cVec, xi_init = xiMat, z = otuTab,
             zero_loc = which( otuTab == 0, arr.ind = TRUE ) - 1,
             Km = Km, log_pK = lp, Mean_vec = mean_prior,
             s2_prior = s2_prior,
             a_ar = a_gamma, b_ar = b_gamma,
             s2_MH_xi = s2_MH,
             gm = theta,
             iter = iter, thin = thin,
             c_mat = c_store, Kp_vec = Kpv, ar_cube = arc,
             K_vec = Kv, eta_mat = eta_store, xi_cube = xic,
             s2a_perform = s2a_perform, s2a_accept = s2a_accept )
  
  ### Return the result
  list( c_MCMC = c_store, Kp_MCMC = Kpv, ar_MCMC = arc,
        K_MCMC = Kv, eta_MCMC = eta_store, xi_MCMC = xic )
  
}

# 79: -------------------------------------------------------------------------
