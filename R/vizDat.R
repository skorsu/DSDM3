## R script for generating the heatmap for the OTU table.
##
## This code uses to generate the visualization of the OTU table.
##
## Updated: May 28, 2025

# Function - Generate the heatmap: --------------------------------------------
vizDat <- function( otuTab ){
  
  n <- dim( otuTab )[ 1 ]
  J <- dim( otuTab )[ 2 ]
  ddat <- data.frame( otuTab, index = 1:n ) %>%
    pivot_longer( !index )
  ddat$name <- factor( ddat$name, levels = paste0( "X", 1:J ) )
  ddat$index <- factor( ddat$index, levels = 1:n )
  
  ggplot( ddat, aes( x = name, y = index, fill = value ) ) +
    geom_tile() +
    theme_bw() +
    scale_fill_gradient( low = "white", high = "#E69F00" ) +
    theme( legend.position = "none",
           axis.text.x = element_blank(),
           axis.ticks.x = element_blank(),
           axis.text.y = element_blank(),
           axis.ticks.y = element_blank() ) +
    labs( x = "Taxa", y = "Observation" ) 
  
}

# 79: -------------------------------------------------------------------------
