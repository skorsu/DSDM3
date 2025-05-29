## R script for helper functions
##
## This file contains functions used for analyzing the results from our model, 
## DSDM3_ZIDM(). It includes functions for creating diagnostic plots, obtaining 
## cluster assignments, and evaluating clustering performance.
##
## Updated: May 28, 2025

# Function: Obtaining the cluster assignment ----------------------------------
final_clus <- function( resultList, burn_in, seed ){
  
  ### Check the required packages
  if( !require( salso ) ){
    stop( "Missing Package: salso" )
  }
  
  suppressWarnings( as.numeric( salso( resultList$c_MCMC[ -( 1:burn_in ), ] ) ) )
  
}

# Function: Diagnostic plots --------------------------------------------------
diag_plot <- function( resultList ){
  
  Km <- ncol( resultList$eta_MCMC )
  iter <- nrow( resultList$eta_MCMC )
  
  ### Kp
  active_clus <- data.frame( Kp = resultList$Kp_MCMC, Iteration = 1:iter ) %>%
    ggplot( aes( x = Iteration, y = Kp ) ) +
    geom_line( linewidth = 0.4 ) +
    scale_y_continuous( limits = c( 1, Km ), breaks = 1:Km ) +
    theme_bw( base_size = 20 ) +
    labs( x = "MCMC Iteration", y = "Number of Active Clusters" ) +
    theme(
      strip.text = element_text( size = 20, face = "bold" ),
      axis.title = element_text( size = 20 ),
      axis.text = element_text( size = 16 ),
      axis.title.x = element_text( vjust = -1 ),
      axis.title.y = element_text( vjust = 2 )
    )
  
  ### Component Weight
  comp_weight <- data.frame( resultList$eta_MCMC ) %>%
    mutate( Iteration = 1:iter ) %>%
    pivot_longer( !Iteration, values_to = "eta", names_to = "Comp" ) %>%
    mutate( Comp = factor( Comp, levels = paste0( "X", 1:Km ), labels = paste0( "Component ", 1:Km ) ) ) %>%
    ggplot( aes( x = Iteration, y = eta ) ) +
    geom_line( linewidth = 0.4 ) +  # match `linewidth` in first plot
    theme_bw( base_size = 20 ) +
    facet_wrap( . ~ Comp ) +
    scale_y_continuous( limits = c( 0, 1 ) ) +
    labs( x = "MCMC Iteration", y = "Component Weight" ) +
    theme(
      strip.text = element_text( size = 20, face = "bold" ),
      axis.title = element_text( size = 20 ),
      axis.text = element_text( size = 16 ),
      axis.title.x = element_text( vjust = -1 ),
      axis.title.y = element_text( vjust = 2 )
    )
  
  list( active_clus = active_clus, comp_weight = comp_weight )
  
}

# Function: ARI ---------------------------------------------------------------
ARI_clus <- function( resultList, truth, burn_in, seed ){

  ### Check the required packages
  if( !require( mclustcomp ) ){
    stop( "Missing Package: mclustcomp" )
  }
  
  c1 <- final_clus( resultList, burn_in, seed )
  suppressWarnings( mclustcomp( c1, truth )[ 1, 2 ] )

}

# 79: -------------------------------------------------------------------------
