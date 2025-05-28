## R script for generating data for a simulation study.
##
## This code generates simulated data with two clusters, each following 
## a zero-inflated Dirichlet-multinomial (ZIDM) mixture model. This simulated 
## data assume that the sequencing depth for all observation are equal and 
## there is no phylogenic tree.
##
## Updated: May 28, 2025

# Function - RA and at-risk probability: --------------------------------------
prob_AR <- function( p, a, baseline ){
  
  baseline + ( ( 1 - baseline ) * ( p ^ a ) )
  
}

# Function - Simulate the signal taxa: ----------------------------------------
data_sim_signal <- function( n1, n2, J1, J2, z_case, 
                             aPhi = 1, bPhi = 1,
                             aLambda = 1, bLambda = 1, 
                             zsum, a = 1, pAR_0 = 0.5,
                             dirConc = 200 ){
  
  ### ( 1 ) Generate the "marginal" RA ( average RA for each taxon )
  pPhi <- rbeta( J1, aPhi, bPhi )
  pLambda <- rbeta( J2, aLambda, bLambda )
  
  ### ( 2 ) Shi's adjustment
  pPhi_A <- ( 1 - ( z_case/5 ) ) * pPhi
  pLambda_A <- ( ( sum( pLambda ) + ( ( z_case/5 ) * sum( pPhi ) ) )/sum( pLambda ) ) * pLambda
  pPhi_B <- ( 1 + ( z_case/5 ) ) * pPhi
  pLambda_B <- ( ( sum( pLambda ) - ( ( z_case/5 ) * sum( pPhi ) ) )/sum( pLambda ) ) * pLambda
  
  ### ( 3 ) Normalize the Shi's adjustment
  unadjust_mar_c1 <- c( pPhi_A, pLambda_A )
  unadjust_mar_c2 <- c( pPhi_B, pLambda_B )
  
  norm_mar <- matrix( NA, nrow = 2, ncol = J1 + J2 )
  norm_mar[ 1, ] <- 0.01 + ( 1 - 0.01 ) * ( unadjust_mar_c1 - min( unadjust_mar_c1 ) ) / ( max( unadjust_mar_c1 ) - min( unadjust_mar_c1 ) )
  norm_mar[ 2, ] <- 0.01 + ( 1 - 0.01 ) * ( unadjust_mar_c2 - min( unadjust_mar_c2 ) ) / ( max( unadjust_mar_c2 ) - min( unadjust_mar_c2 ) )

  ### ( 4 ) Calculate at-risk probability from the "marginal"
  ar_mat_clus <- matrix( NA, nrow = 2, ncol = J1 + J2 )
  ar_mat_clus[ 1, ] <- sapply( norm_mar[ 1, ], function( x ){ prob_AR( x, a, pAR_0 ) } )
  ar_mat_clus[ 2, ] <- sapply( norm_mar[ 2, ], function( x ){ prob_AR( x, a, pAR_0 ) } )
  
  ### ( 5 ) Generate the OTU table
  clus <- c( rep( 1, n1 ), rep( 2, n2 ) )
  dat <- matrix( NA, ncol = J1 + J2, nrow = n1 + n2 )
  ar <- matrix( NA, ncol = J1 + J2, nrow = n1 + n2 )
  
  for( i in 1:( n1 + n2 ) ){

    gm <- rbinom( J1 + J2, 1, ar_mat_clus[ clus[ i ], ] )
    ar[ i, ] <- gm
    prob <- rdirichlet( 1, dirConc * norm_mar[ clus[ i ], ] / sum( norm_mar[ clus[ i ], ] ) )
    dat[ i, ] <- rmultinom( 1, zsum, ( gm * prob )/sum( gm * prob ) )

  }
  
  list( dat = dat, ar = ar, clus = clus )
  
}

# Function - Simulate the noise taxa: -----------------------------------------
data_sim_noise <- function( n, J, aBeta = 1, bBeta = 1, zsum, a = 1, 
                            pAR_0 = 0.5, dirConc = 200 ){
  
  ### ( 1 ) Generate the "marginal" RA ( average RA for each taxon )
  RA <- rbeta( J, aBeta, bBeta )
  
  ### ( 2 ) Calculate at-risk probability from the "marginal"
  pAR <- sapply( RA, function( x ){ prob_AR( x, a, pAR_0 ) } )
  
  ### ( 3 ) Generate the OTU table
  dat <- matrix( NA, ncol = J, nrow = n )
  ar <- matrix( NA, ncol = J, nrow = n )
  
  for( i in 1:n ){
    
    gm <- rbinom( J, 1, pAR )
    ar[ i, ] <- gm
    pTaxon <- rdirichlet( 1, dirConc * RA/sum( RA ) )
    dat[ i, ] <- rmultinom( 1, zsum, ( gm * pTaxon )/sum( gm * pTaxon ) )
    
  }
  
  list( dat = dat, ar = ar )
  
}

# Function: Combining signal and noise taxon: ---------------------------------
cluster_dat <- function( n1, n2, Jnoise, Jsignal_c1, Jsignal_c2,
                         z_case = 2.5, 
                         z_sum_noise, z_sum_signal,
                         dirConc_noise, dirConc_signal,
                         a_noise, a_signal,
                         pAR_0_noise, pAR_0_signal ){
  
  
  ### ( 1 ) Run the function for simulating signal and noise taxa
  signal_taxa <- data_sim_signal( n1, n2, Jsignal_c1, Jsignal_c2, z_case, 
                                  aPhi = 1, bPhi = 1, aLambda = 1, bLambda = 1, 
                                  zsum = z_sum_signal, a = a_signal, 
                                  pAR_0 = pAR_0_signal, dirConc = dirConc_signal )
  noise_taxa <- data_sim_noise( n = n1 + n2, J = Jnoise, aBeta = 1, 
                                bBeta = 1, zsum = z_sum_noise, a = a_noise, 
                                pAR_0 = pAR_0_noise, dirConc = dirConc_noise )
  
  ### ( 2 ) Combine
  dat <- cbind( signal_taxa$dat, noise_taxa$dat )
  ar <- cbind( signal_taxa$ar, noise_taxa$ar )
  
  list( dat = dat, ar = ar, clus = signal_taxa$clus )
  
}


# Function: User-friendly function: -------------------------------------------
sim_ZIDMclus <- function( n1, n2, Jnoise, Jsignal, z_sum_noise, z_sum_signal,
                          a, pAR0, seed ){
  
  set.seed( seed = seed )
  cluster_dat( n1 = n1, n2 = n2, Jnoise = Jnoise, Jsignal_c1 = Jsignal/2, 
               Jsignal_c2 = Jsignal/2, z_case = 2.5, 
               z_sum_noise = z_sum_noise, z_sum_signal = z_sum_signal, 
               dirConc_noise = 200, dirConc_signal = 200,
               a_noise = a, a_signal = a, 
               pAR_0_noise = pAR0, pAR_0_signal = pAR0 )
  
}

# 79: -------------------------------------------------------------------------
