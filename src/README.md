{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Code Update and To-Do List\
\
## Update\
\
### General Updates\
- **C++ Coding Style**: Implement C++ reference style coding across all functions to ensure consistency and improve maintainability.\
\
### Sampling Updates\
- **At-Risk Indicator ($\\gamma_\{ij\}$)**:\
    - Modified the approach to handle zero counts by creating a matrix of the index of zero counts and looping through this matrix. This method reduces computational costs compared to the previous nested loop structure.\
\
- **Cluster Concentration ($\\xi_\{kj\}$)**:\
    - Adopted the sequential update method for the covariance matrix to minimize computational costs and processing time (Reference: [Sequential Update Method](https://stats.stackexchange.com/a/310701)).\
    - Adjusted the adaptive Metropolis-Hastings (MH) approach to trigger only after a cluster has existed for `$t_\{thres\}$` consecutive iterations, enhancing the algorithm's efficiency and relevance.\
\
- **Joint Update of $\\xi_\{kj\}$ and $c_\{I\}$**:\
    - Implemented the Split-Merge technique as described by Jain 2007.\
\
- **Cluster Assignment ($c_\{i\}$)**:\
    - Integrated Algorithm 8 from Neal 2000 to update cluster assignments, aiming for more robust and accurate clustering results.\
\
## To-Do\
- **Simulations and Applications**:\
    - Rerun the updated code on both simulation studies and application data to validate changes and assess improvements.\
\
\
}