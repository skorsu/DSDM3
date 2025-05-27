# A Bayesian Semiparametric Mixture Model for Clustering Zero-Inflated Microbiome Data

This repository contains the `DSDM3` package and supporting code for fitting a Mixture of Finite Mixtures (MFM) model tailored to zero-inflated microbiome count data.

The core computational functions are implemented in C++ using the `Rcpp` and `RcppArmadillo` packages. These functions, together with those for simulating synthetic data used in the simulation study and performing inference, are complemented by R wrapper functions that facilitate integration into the R environment.

To use the `DSDM3` package, make sure the following R packages are installed using the `install.packages()` function:

* `devtools`
* `Rcpp`
* `RcppArmadillo`
* `dirmult`
* `mclustcomp`
* `salso`

To install the `DSDM3` package from GitHub, run the following command in the R environment:

```r
devtools::install_github("skorsu/DSDM3")
```