#include "RcppArmadillo.h"

// [[Rcpp::depends(RcppArmadillo)]]

#define pi 3.141592653589793238462643383280

// [[Rcpp::export]]
arma::vec rmultinom_1(arma::vec &probs_arma){
  
  /* Description: sample from the multinomial(N, probs).
   * Credit: https://gallery.rcpp.org/articles/recreating-rmultinom-and-rpois-with-rcpp/
   */
  
  Rcpp::NumericVector probs = Rcpp::NumericVector(probs_arma.begin(), probs_arma.end());
  unsigned int N = probs_arma.size();
  Rcpp::IntegerVector outcome(N);
  rmultinom(1, probs.begin(), N, outcome.begin());
  arma::vec mul_result = Rcpp::as<arma::vec>(Rcpp::wrap(outcome));

  return mul_result;
  
}

// [[Rcpp::export]]
arma::vec log_sum_exp(arma::vec log_unnorm_prob){
  
  /* Description: This function will calculate the normalized probability 
   *              by applying log-sum-exp trick on the log-scale probability.
   * Credit: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
   */
  
  double max_elem = log_unnorm_prob.max();
  double t = log(0.00000000000000000001) - log(log_unnorm_prob.size());          
  
  for(int k = 0; k < log_unnorm_prob.size(); ++k){
    double prob_k = log_unnorm_prob.at(k) - max_elem;
    if(prob_k > t){
      log_unnorm_prob.row(k).fill(std::exp(prob_k));
    } else {
      log_unnorm_prob.row(k).fill(0.00000000000000000001);
    }
  } 
  
  // Normalize the vector
  return log_unnorm_prob/arma::accu(log_unnorm_prob);
}

// [[Rcpp::export]]
double log_marginal(arma::vec zi, arma::vec gmi, arma::vec beta_k){
  
  /* Calculate the proportional of the marginal probability in a log scale. */
  
  double result = 0.0;
  
  arma::uvec gm1 = arma::find(gmi == 1);
  arma::vec xi_k = arma::exp(beta_k);
  
  result += std::lgamma(arma::accu(xi_k.rows(gm1)));
  result -= arma::accu(arma::lgamma(xi_k.rows(gm1)));
  result += arma::accu(arma::lgamma(zi.rows(gm1) + xi_k.rows(gm1)));
  result -= std::lgamma(arma::accu(zi.rows(gm1) + xi_k.rows(gm1)));
  
  return result;
  
}

// [[Rcpp::export]]
double logmar_ik(arma::rowvec zi, arma::rowvec atrisk_i, arma::rowvec beta_k){
  
  /* Calculate the log marginal for each observation in a log scale for the 
   * observation i, given that it is in the cluster k */
  
  double lmar = 0.0;
  
  lmar += std::lgamma(arma::accu(zi) + 1);
  lmar -= arma::accu(arma::lgamma(zi + 1));
  
  arma::rowvec e_beta = arma::exp(beta_k); 
  arma::rowvec zi_e_beta = zi + e_beta;
  arma::uvec atrisk_one = arma::find(atrisk_i == 1);
  
  lmar += std::lgamma(arma::accu(e_beta.cols(atrisk_one)));
  lmar -= arma::accu(arma::lgamma(e_beta.cols(atrisk_one)));
  
  lmar += arma::accu(arma::lgamma(zi_e_beta.cols(atrisk_one)));
  lmar -= std::lgamma(arma::accu(zi_e_beta.cols(atrisk_one)));
  
  return lmar;
  
}

// [[Rcpp::export]]
arma::vec logmar_k(arma::mat z, arma::mat atrisk, arma::rowvec beta_k){
  
  /* Calculate the log marginal for each observation in a log scale, given 
     that they are all in cluster k */
  
  arma::vec lmar(z.n_rows, arma::fill::zeros);
  arma::rowvec e_beta = arma::exp(beta_k);
  
  // At-risk indicator always equal 1 for non-zero z_ijk. It can be 0 if z_ijk = 0 only.
  // Data (z)
  lmar += arma::lgamma(arma::sum(z, 1) + 1);
  lmar -= arma::sum(arma::lgamma(z + 1), 1);
  
  // Hyperparameter part (beta)
  for(int i = 0; i < z.n_rows; ++i){
    arma::uvec atrisk_zero = arma::find(atrisk.row(i) == 1);
    arma::rowvec zi_ebeta = z.row(i) + e_beta;
    lmar[i] += std::lgamma(arma::accu(e_beta.cols(atrisk_zero)));
    lmar[i] -= arma::accu(arma::lgamma(e_beta.cols(atrisk_zero)));
    lmar[i] += arma::accu(arma::lgamma(zi_ebeta.cols(atrisk_zero)));
    lmar[i] -= std::lgamma(arma::accu(zi_ebeta.cols(atrisk_zero)));
  }
  
  return lmar;
  
}

// [[Rcpp::export]]
arma::mat logmar(arma::mat z, arma::mat atrisk, arma::mat beta_mat){
  
  /* Calculate the log marginal for all observations in all possible clusters */
  unsigned int Kmax = beta_mat.n_rows;
  arma::mat logmar_mat(z.n_rows, Kmax, arma::fill::zeros);
  
  // At-risk indicator always equal 1 for non-zero z_ijk. It can be 0 if z_ijk = 0 only.
  // Data (z)
  arma::vec sum_zi = arma::sum(z, 1); 
  
  logmar_mat += arma::repelem(arma::lgamma(sum_zi + 1), 1, Kmax);
  logmar_mat -= arma::repelem(arma::sum(arma::lgamma(z + 1), 1), 1, Kmax);
  
  // Cluster (Beta) -- Loop through every cluster for changing the beta
  for(int k = 0; k < Kmax; ++k){
    arma::rowvec e_beta_k = arma::exp(beta_mat.row(k));
    arma::vec marginal_part(z.n_rows, arma::fill::zeros);
    
    for(int i = 0; i < z.n_rows; ++i){
      arma::uvec atrisk_loc = arma::find(atrisk.row(i) == 1);
      arma::rowvec e_beta_k_zi = z.row(i) + e_beta_k;
      marginal_part[i] += std::lgamma(arma::accu(e_beta_k.cols(atrisk_loc)));
      marginal_part[i] -= arma::accu(arma::lgamma(e_beta_k.cols(atrisk_loc)));
      marginal_part[i] += arma::accu(arma::lgamma(e_beta_k_zi.cols(atrisk_loc)));
      marginal_part[i] -= std::lgamma(arma::accu(e_beta_k_zi.cols(atrisk_loc))); 
    }
    
    logmar_mat.col(k) += marginal_part;
    
  }
  
  return logmar_mat;
    
}

// [[Rcpp::export]]
double log_atrisk(arma::rowvec zi, arma::rowvec atrisk_i, arma::rowvec beta_k,
                  double r0g, double r1g){
  
  double result = logmar_ik(zi, atrisk_i, beta_k);
  arma::uvec zi_zero = arma::find(zi == 0);
  arma::rowvec ar_zi_zero = atrisk_i.cols(zi_zero);
  result += arma::accu(arma::lgamma(ar_zi_zero + r0g));
  result += arma::accu(arma::lgamma((1 - ar_zi_zero) + r1g));
  result -= (ar_zi_zero.size() * std::lgamma(r0g + r1g + 1));
  
  return result;
  
}

// [[Rcpp::export]]
Rcpp::List launch_mcmc(arma::mat z, arma::mat atrisk, arma::mat beta_mat, 
                       arma::uvec ci_old, unsigned int launch_iter, arma::uvec S, 
                       arma::uvec samp_clus, arma::vec nk){
  
  /* This algorithm performs the launch step. */
  
  arma::uvec ci_result(ci_old);
  
  // Calculate the log marginal for the cluster of interest.
  arma::mat lmar_sc(z.n_rows, 2, arma::fill::zeros);
  lmar_sc.col(0) = logmar_k(z, atrisk, beta_mat.row(samp_clus[0]));
  lmar_sc.col(1) = logmar_k(z, atrisk, beta_mat.row(samp_clus[1]));
  
  // Perform a restricted reallocation step
  for(int t = 0; t < launch_iter; ++t){
    for(int ss = 0; ss < S.size(); ++ss){
      int s = S[ss];
      nk[ci_result[s]] -= 1;
      
      // Change of index: From Kmax to the samp_clus only.
      arma::vec nk_sc = nk.rows(samp_clus);
      arma::vec log_realloc_prob = arma::log(nk_sc) + lmar_sc.row(s).t();
      arma::vec realloc_prob = log_sum_exp(log_realloc_prob);
      arma::vec ck_new_kk_vec = rmultinom_1(realloc_prob);
      arma::uvec kk_new_vec = arma::find(ck_new_kk_vec == 1);
      int kk_new = kk_new_vec[0];
      
      // Need to check the index back: From active to Kmax.
      int new_ci = samp_clus[kk_new];
      nk[new_ci] += 1;
      ci_result[s] = new_ci;
      
    }
  }
  
  Rcpp::List result;
  result["ci_result"] = ci_result;
  result["nk"] = nk;
  return result;
  
}

// [[Rcpp::export]]
double log_proposal(arma::mat z, arma::mat atrisk, arma::mat beta_mat, 
                    arma::uvec ci_A, arma::uvec ci_B, arma::uvec S, 
                    arma::uvec samp_clus){
  
  /* This function will calculate the proposal distribution q(ci_A|ci_B) in
     a log scale. */
  
  double lq = 0.0;
  
  // Calculate the log marginal for the interested clusters, samp_clus.
  arma::mat lmar(z.n_rows, 2, arma::fill::zeros);
  lmar.col(0) = logmar_k(z, atrisk, beta_mat.row(samp_clus[0]));
  lmar.col(1) = logmar_k(z, atrisk, beta_mat.row(samp_clus[1]));
  
  // Calculate the initial nk
  arma::vec nk(2, arma::fill::zeros);
  for(int kk = 0; kk < 2; ++kk){
    nk[kk] += arma::accu(ci_B == samp_clus[kk]);
  }
  
  // Calculate the log proposal
  arma::uvec ci_int(ci_B);
  for(int ss = 0; ss < S.size(); ++ss){
    int s = S[ss];
    
    // Index Change: From Kmax to samp_clus
    arma::uvec ciB_index = arma::find(samp_clus == ci_B[s]);
    arma::uvec ciA_index = arma::find(samp_clus == ci_A[s]);
    
    nk[ciB_index[0]] -= 1;
    arma::vec alloc_prob = log_sum_exp(arma::log(nk) + lmar.row(s).t());
    lq += std::log(alloc_prob[ciA_index[0]]);
    ci_int[s] = ci_A[s];
    nk[ciA_index[0]] += 1;
    
  } 
  
  return lq;
  
}

// [[Rcpp::export]]
arma::mat split_beta(unsigned int nbeta, unsigned int clus_split, 
                     unsigned int clus_new, arma::mat beta_old, 
                     double mu, double s2){
  
  arma::mat beta_new(beta_old);
  
  // First, propose a new beta vector from the cluster that we want to split.
  arma::vec beta_proposed = beta_old.row(clus_split).t();
  
  // Sample the index that we want to update beta
  arma::uvec beta_index = arma::randperm(beta_proposed.size(), nbeta);
  arma::vec samp_bt = arma::mvnrnd(mu * arma::ones(nbeta),
                                   s2 * arma::eye(nbeta, nbeta));
  
  beta_proposed.rows(beta_index) = samp_bt;
  beta_new.row(clus_new) = beta_proposed.t();
  
  return beta_new;
  
}

// Algorithms for updating parameters: -----------------------------------------

// At-risk
// [[Rcpp::export]]
arma::mat update_atrisk(arma::mat z, arma::mat atrisk_old, arma::mat beta_mat, 
                        arma::uvec ci, double r0g, double r1g){

  arma::mat beta_k_mat = beta_mat.rows(ci);
  arma::mat atrisk_new(atrisk_old);
  
  for(int i = 0; i < z.n_rows; ++i){
    
    arma::rowvec zi = z.row(i);
    arma::rowvec beta_i = beta_k_mat.row(i);
    arma::uvec zi_zero = arma::find(zi == 0);
    
    for(int jj = 0; jj < zi_zero.size(); ++jj){
      
      // Create a proposed at-risk vector 
      arma::rowvec old_ar = atrisk_new.row(i);
      arma::rowvec proposed_ar(old_ar);
      proposed_ar[zi_zero[jj]] = 1 - old_ar[zi_zero[jj]];
      
      // Calculate logA
      double logA = 0.0;
      logA += log_atrisk(zi, proposed_ar, beta_i, r0g, r1g);
      logA -= log_atrisk(zi, old_ar, beta_i, r0g, r1g);
      
      // Accept the proposed at-risk
      double logU = std::log(R::runif(0.01, 1.0));
      if(logU < logA){
        atrisk_new.row(i) = proposed_ar;
      }
      
    }
  
  }
  
  return atrisk_new;
  
}

// Beta
// [[Rcpp::export]]
Rcpp::List update_beta(arma::mat z, arma::mat atrisk, arma::mat beta_old, 
                       arma::uvec ci, double mu, double s2, double s2_MH){
  
  /* Update the beta vector parameters (Parameter of information sharing 
     within the cluster) - We update only for the active clusters */
  
  arma::vec accept_MH(beta_old.n_rows, arma::fill::value(-1));
  
  // Active Clusters
  arma::uvec active_clus = arma::unique(ci);
  unsigned int Kpos = active_clus.size();
  
  // Update the beta vector for the active cluster.
  arma::mat beta_result(beta_old.n_rows, beta_old.n_cols, arma::fill::zeros);
  for(int kk = 0; kk < Kpos; ++kk){
    
    int k = active_clus[kk];
    arma::uvec ck = arma::find(ci == k);
    arma::rowvec bk_old = beta_old.row(k);
    
    // Proposed a new beta_k
    arma::rowvec bk_pro = arma::conv_to<arma::rowvec>::from(arma::mvnrnd(bk_old.t(), 
                                                                         s2_MH * arma::eye(z.n_cols, z.n_cols)));
    // Calculate logA
    double logA = 0.0;
    logA += arma::accu(arma::log_normpdf(bk_pro, mu, std::sqrt(s2)));
    logA += arma::accu(logmar_k(z, atrisk, bk_pro).rows(ck));
    logA -= arma::accu(arma::log_normpdf(bk_old, mu, std::sqrt(s2)));
    logA -= arma::accu(logmar_k(z, atrisk, bk_old).rows(ck));
    
    double logU = std::log(R::runif(0.01, 1.0));
    if(logU < logA){
      beta_result.row(k) = bk_pro;
      accept_MH.row(k).fill(1);
    } else {
      beta_result.row(k) = bk_old;
      accept_MH.row(k).fill(0);
    }
    
  }
  
  Rcpp::List result; 
  result["beta_update"] = beta_result;
  result["accept_MH"] = accept_MH;
  return result;
  
}

// [[Rcpp::export]]
Rcpp::List update_beta_adaptive(unsigned int t, unsigned int t_threshold, 
                                arma::mat z, arma::mat atrisk, arma::cube beta_record, 
                                arma::mat beta_init, arma::uvec ci, double mu, double s2, double s2_MH){
  
  /* Update the beta vector parameters (Parameter of information sharing 
   within the cluster) - We update only for the active clusters */
  
  arma::vec accept_MH(beta_record.n_rows, arma::fill::value(-1));
  arma::mat beta_old = beta_init;
  if(t > 0){
    beta_old = beta_record.slice(t - 1);
  }
  double cd = std::pow(2.4, 2.0)/beta_old.n_cols;
  
  // Active Clusters
  arma::uvec active_clus = arma::unique(ci);
  unsigned int Kpos = active_clus.size();
  
  // Update the beta vector for the active cluster.
  arma::mat beta_result(beta_old.n_rows, beta_old.n_cols, arma::fill::zeros);
  for(int kk = 0; kk < Kpos; ++kk){
    
    int k = active_clus[kk];
    arma::uvec ck = arma::find(ci == k);
    arma::rowvec bk_old = beta_old.row(k);
    
    // Proposed a new beta_k
    arma::mat covM = s2_MH * arma::eye(z.n_cols, z.n_cols);
    
    if(t > t_threshold){
      covM *= (std::pow(1, -10) * cd);
      covM += (cd * arma::cov(beta_record.head_slices(t + 1).row_as_mat(k)));
    }
    
    arma::rowvec bk_pro = arma::conv_to<arma::rowvec>::from(arma::mvnrnd(bk_old.t(), covM));
    
    // Calculate logA
    double logA = 0.0;
    logA += arma::accu(arma::log_normpdf(bk_pro, mu, std::sqrt(s2)));
    logA += arma::accu(logmar_k(z, atrisk, bk_pro).rows(ck));
    logA -= arma::accu(arma::log_normpdf(bk_old, mu, std::sqrt(s2)));
    logA -= arma::accu(logmar_k(z, atrisk, bk_old).rows(ck));
    
    double logU = std::log(R::runif(0.01, 1.0));
    if(logU < logA){
      beta_result.row(k) = bk_pro;
      accept_MH.row(k).fill(1);
    } else { 
      beta_result.row(k) = bk_old;
      accept_MH.row(k).fill(0);
    } 
    
  } 
  
  Rcpp::List result; 
  result["beta_update"] = beta_result;
  result["accept_MH"] = accept_MH;
  return result;
  
}

// Cluster Assignment: Reallocation
// [[Rcpp::export]]
arma::uvec realloc_full(unsigned int Kmax, arma::mat z, arma::mat atrisk, 
                        arma::mat beta_mat, arma::uvec ci_old, double theta){
  
  arma::uvec ci_updated(ci_old);
  arma::uvec active_clus = arma::unique(ci_old);
  unsigned int Kpos = active_clus.size();
  
  // Note that we allow the observation to go to the active cluster only
  
  // Calculate the log marginal for every observation in each possible clusters.
  // We can do this as this step does not change beta vector.
  arma::mat lmar_realloc = logmar(z, atrisk, beta_mat.rows(active_clus));
  
  // Find the number of the observations in each active cluster
  arma::vec nkk(Kpos, arma::fill::zeros);
  for(int kk = 0; kk < Kpos; ++kk){
    int k = active_clus[kk];
    nkk[kk] += arma::accu(ci_old == k);
  } 
  
  for(int i = 0; i < z.n_rows; ++i){
    int ci_old = ci_updated[i];
    
    // Adjust from Kmax space to Kpos space
    nkk.elem(arma::find(active_clus == ci_old)) -= 1; 
    arma::vec log_realloc_prob = arma::log(nkk + theta) + lmar_realloc.row(i).t();
    arma::vec realloc_prob = log_sum_exp(log_realloc_prob);
    arma::vec ck_new_vec = rmultinom_1(realloc_prob);
    arma::uvec kk_new_vec = arma::find(ck_new_vec == 1);
    int kk_new_ci = kk_new_vec[0];
    nkk[kk_new_ci] += 1;
    
    // Adjust from Kpos space back to Kmax space
    ci_updated[i] = active_clus[kk_new_ci];
    
  }
  
  return ci_updated;
  
}

// Cluster Assignment: Split-Merge
// [[Rcpp::export]]
Rcpp::List sm(unsigned int Kmax, unsigned int nbeta_split, 
              arma::mat z, arma::mat atrisk, arma::mat beta_old, 
              arma::uvec ci_old, double theta, 
              double mu, double s2, unsigned int launch_iter,
              double r0c, double r1c){
  
  int split_index = -1;
  arma::uvec active_clus = arma::unique(ci_old); 
  unsigned int Kpos = active_clus.size();
  
  // Find the number of the observation in each cluster
  arma::vec nk(Kmax, arma::fill::zeros);
  for(int k = 0; k < Kmax; ++k){
    nk[k] += arma::accu(ci_old == k);
  }
  
  arma::vec nk_old(nk);
  
  // Start with determine to split (expand) or merge (collapse)
  // samp_ind: the index of two observations used for considering to split or merge.
  // samp_clus: the cluster of the two observations from samp_ind.
  
  arma::uvec samp_ind = arma::randperm(z.n_rows, 2);
  while((Kpos == Kmax) and 
          (ci_old[samp_ind[0]] == ci_old[samp_ind[1]])){
    samp_ind = arma::randperm(z.n_rows, 2);
  }
  arma::uvec samp_clus = ci_old.rows(samp_ind);
  
  // Then, create a set S: the index of the observation with the same cluster as
  // samp_ind, but not the samp_ind itself.
  
  arma::uvec S = arma::find(ci_old == samp_clus[0] or ci_old == samp_clus[1]);
  arma::uvec notS = arma::find(S != samp_ind[0] and S != samp_ind[1]);
  S = S.rows(notS);
  
  // Perform the launch step
  arma::uvec ci_launch(ci_old);
  arma::mat beta_launch(beta_old);
  
  if(samp_clus[0] == samp_clus[1]){
    // If split, proposed a new cluster with a new beta vector simulated from the prior.
    split_index = 1;
    arma::uvec inactive_clus = arma::find(nk == 0);
    arma::uvec new_active_index = arma::randperm(inactive_clus.size(), 1);
    arma::uvec new_active_clus = inactive_clus.row(new_active_index[0]);
    nk[samp_clus[0]] -= 1;
    nk[new_active_clus[0]] += 1;
    samp_clus[0] = new_active_clus[0];
    ci_launch[samp_ind[0]] = new_active_clus[0];
    beta_launch = split_beta(nbeta_split, samp_clus[1], samp_clus[0], beta_old, mu, s2);
  } else {
    split_index = 0;
  }
  
  // Randomly assign S to one of the samp_clus.
  for(int ss = 0; ss < S.size(); ++ss){
    int s = S[ss];
    int new_clus_index = std::round(R::runif(0.0, 1.0));
    nk[ci_launch[s]] -= 1;
    ci_launch[s] = samp_clus[new_clus_index];
    nk[ci_launch[s]] += 1;
  }
  
  // Perform Launch Iteration
  Rcpp::List launch_result = launch_mcmc(z, atrisk, beta_launch, ci_launch, 
                                         launch_iter, S, samp_clus, nk);
  arma::uvec ll_ci = launch_result["ci_result"];
  arma::vec nnk = launch_result["nk"];
  ci_launch = ll_ci;
  nk = nnk;
  
  // Proposed a new cluster assignment
  arma::uvec ci_proposed(ci_launch);
  if(split_index == 1){
    // If we split, we will do one final launch iteration.
    Rcpp::List proposed_result = launch_mcmc(z, atrisk, beta_launch, ci_proposed, 
                                             launch_iter, S, samp_clus, nk);
    arma::uvec pp_ci = proposed_result["ci_result"];
    arma::vec nnnk = proposed_result["nk"];
    ci_proposed = pp_ci;
    nk = nnnk;
  } else {
    nk[samp_clus[1]] += nk[samp_clus[0]];
    nk[samp_clus[0]] = 0;
    ci_proposed.rows(S).fill(samp_clus[1]);
    ci_proposed.rows(samp_ind).fill(samp_clus[1]);
  }
  
  // MH
  double logA = 0.0;
  
  // Data Part -- Consider only the observation in set S and samp_ind.
  for(int ss = 0; ss < S.size(); ++ss){
    int s = S[ss];
    logA += logmar_ik(z.row(s), atrisk.row(s), beta_launch.row(ci_proposed[s]));
    logA -= logmar_ik(z.row(s), atrisk.row(s), beta_old.row(ci_old[s]));
  }
  
  for(int ii = 0; ii < 2; ++ii){
    int i = samp_ind[ii];
    logA += logmar_ik(z.row(i), atrisk.row(i), beta_launch.row(ci_proposed[i]));
    logA -= logmar_ik(z.row(i), atrisk.row(i), beta_old.row(ci_old[i]));
  }
  
  // Cluster Dirichlet Part
  // Proposed
  arma::uvec active_proposed = arma::find(nk == 0);
  logA += std::lgamma(theta * active_proposed.size());
  logA -= active_proposed.size()  * std::lgamma(theta);
  arma::vec ntheta_proposed = theta + nk;
  logA += arma::accu(arma::lgamma(ntheta_proposed.rows(active_proposed)));
  logA -= std::lgamma(arma::accu(ntheta_proposed.rows(active_proposed)));
  
  // Old
  arma::uvec active_old = arma::find(nk_old == 0);
  logA += std::lgamma(theta * active_old.size());
  logA -= active_old.size()  * std::lgamma(theta);
  arma::vec ntheta_old = theta + nk_old;
  logA += arma::accu(arma::lgamma(ntheta_old.rows(active_old)));
  logA -= std::lgamma(arma::accu(ntheta_old.rows(active_old)));
  
  // Cluster Beta Distribution and the Beta
  if(split_index == 1){
    logA += std::log(r0c);
    logA -= std::log(r1c);
    logA += arma::accu(arma::log_normpdf(beta_launch.row(samp_clus[0]), mu, std::sqrt(s2)));
  } else {
    logA += std::log(r1c);
    logA -= std::log(r0c);
    logA -= arma::accu(arma::log_normpdf(beta_launch.row(samp_clus[0]), mu, std::sqrt(s2)));
  }
  
  // Proposal
  logA += log_proposal(z, atrisk, beta_launch, ci_launch, ci_proposed, S, samp_clus);
  if(split_index == 1){
    logA -= log_proposal(z, atrisk, beta_launch, ci_proposed, ci_launch, S, samp_clus);
  }
  
  // MH: Final Step
  arma::uvec ci_result;
  arma::mat beta_result;
  int accept_SM = 0;
  
  double logU = std::log(R::runif(0.0, 1.0));
  if(logU < logA){
    accept_SM += 1;
    ci_result = ci_proposed;
    beta_result = beta_launch;
  } else {
    // Perform only reallocation step
    ci_result = ci_old;
    beta_result = beta_old;
  }
  
  Rcpp::List result;
  result["split_index"] = split_index;
  result["logA"] = logA;
  result["accept_SM"] = accept_SM;
  result["ci_result"] = ci_result;
  result["beta_result"] = beta_result;
  return result;
  
}

// Combined Function: ----------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List mod(unsigned int iter, unsigned int Kmax, unsigned int nbeta_split, 
               arma::mat z, arma::mat atrisk_init, arma::mat beta_init, 
               arma::uvec ci_init, double theta, double mu, double s2, 
               double s2_MH, unsigned int launch_iter, 
               double r0g, double r1g, double r0c, double r1c, 
               unsigned int thin){
  
  std::cout << "Total Iteration: " << iter << std::endl;
  std::cout << "Thinning: " << thin << std::endl;
  std::cout << "Total Recorded Iteration: " << iter/thin << std::endl;
  std::cout << "############ PERFORM THE CLUSTERING ############" << std::endl;
  unsigned int save_col = 0;
  
  arma::umat ci_result(z.n_rows, iter/thin, arma::fill::value(Kmax + 1));
  arma::mat MH_accept(iter/thin, Kmax, arma::fill::value(-2));
  arma::cube beta_result(Kmax, z.n_cols, iter/thin, arma::fill::value(0));
  arma::cube atrisk_result(z.n_rows, z.n_cols, iter/thin, arma::fill::value(0));
  arma::vec sm_status(iter, arma::fill::value(-2));
  arma::vec sm_accept(iter, arma::fill::value(-2));
  
  arma::mat atrisk_mcmc(atrisk_init);
  arma::mat beta_mcmc(beta_init); 
  Rcpp::List sm_sum;
  Rcpp::List beta_sum; 
  arma::uvec ci_mcmc(ci_init);
  
  for(int t = 0; t < iter; ++t){
    // update at-risk
    atrisk_mcmc = update_atrisk(z, atrisk_init, beta_init, ci_init, r0g, r1g);
    // update beta
    beta_sum = update_beta(z, atrisk_mcmc, beta_init, ci_init, mu, s2, s2_MH);
    arma::mat bMCMC = beta_sum["beta_update"];
    // reallocation
    ci_mcmc = realloc_full(Kmax, z, atrisk_mcmc, bMCMC, ci_init, theta);
    // sm
    sm_sum = sm(Kmax, nbeta_split, z, atrisk_mcmc, bMCMC, ci_mcmc, 
                theta, mu, s2, launch_iter, r0c, r1c);
    
    // record the result
    atrisk_init = atrisk_mcmc;
    arma::mat beta_final = sm_sum["beta_result"];
    beta_init = beta_final;
    
    arma::uvec ci_final = sm_sum["ci_result"];
    ci_init = ci_final;
    
    sm_status[t] = sm_sum["split_index"];
    sm_accept[t] = sm_sum["accept_SM"];
    
    if(((t + 1) - (floor((t + 1)/thin) * thin)) == 0){
      std::cout << "Iter: " << (t+1) << " - Done!" << std::endl;
      atrisk_result.slice(save_col) = atrisk_init;
      beta_result.slice(save_col) = beta_final;
      ci_result.col(save_col) = ci_final;
      arma::vec MHa = beta_sum["accept_MH"];
      MH_accept.row(save_col) = MHa.t();
      save_col += 1;
    }
    
  }
  
  Rcpp::List result;
  result["ci_result"] = ci_result.t();
  result["beta_result"] = beta_result;
  result["atrisk_result"] = atrisk_result;
  result["sm_status"] = sm_status;
  result["sm_accept"] = sm_accept;
  result["MH_accept"] = MH_accept;
  return result;
  
}

// [[Rcpp::export]]
Rcpp::List mod_adaptive(unsigned int iter, unsigned int Kmax, unsigned int nbeta_split, 
                        arma::mat z, arma::mat atrisk_init, arma::mat beta_init, 
                        arma::uvec ci_init, double theta, double mu, double s2, 
                        double s2_MH, unsigned int t_thres, unsigned int launch_iter, 
                        double r0g, double r1g, double r0c, double r1c, 
                        unsigned int thin){
  
  std::cout << "Total Iteration: " << iter << std::endl;
  std::cout << "Thinning: " << thin << std::endl;
  std::cout << "Total Recorded Iteration: " << iter/thin << std::endl;
  std::cout << "############ PERFORM THE CLUSTERING ############" << std::endl;
  unsigned int save_col = 0;
  
  arma::umat ci_result(z.n_rows, iter/thin, arma::fill::value(Kmax + 1));
  arma::cube atrisk_result(z.n_rows, z.n_cols, iter/thin, arma::fill::value(0));
  arma::vec sm_status(iter, arma::fill::value(-2));
  arma::vec sm_accept(iter, arma::fill::value(-2));
  
  arma::mat MH_accept(iter, Kmax, arma::fill::value(-2));
  arma::cube beta_result(Kmax, z.n_cols, iter, arma::fill::value(0));
  
  arma::mat beta0 = beta_init; 
  arma::mat atrisk_mcmc(atrisk_init);
  arma::mat beta_mcmc(beta_init); 
  Rcpp::List sm_sum;
  Rcpp::List beta_sum; 
  arma::uvec ci_mcmc(ci_init);
  
  for(int t = 0; t < iter; ++t){
    // update at-risk
    atrisk_mcmc = update_atrisk(z, atrisk_init, beta_init, ci_init, r0g, r1g);
    // update beta
    beta_sum = update_beta_adaptive(t, t_thres, z, atrisk_mcmc, beta_result, beta0,
                                    ci_init, mu, s2, s2_MH);
    arma::mat bMCMC = beta_sum["beta_update"];
    // reallocation
    ci_mcmc = realloc_full(Kmax, z, atrisk_mcmc, bMCMC, ci_init, theta);
    // sm
    sm_sum = sm(Kmax, nbeta_split, z, atrisk_mcmc, bMCMC, ci_mcmc, 
                theta, mu, s2, launch_iter, r0c, r1c);
    
    // record the result
    atrisk_init = atrisk_mcmc;
    arma::mat beta_final = sm_sum["beta_result"];
    beta_init = beta_final;
    
    arma::uvec ci_final = sm_sum["ci_result"];
    ci_init = ci_final;
    
    sm_status[t] = sm_sum["split_index"];
    sm_accept[t] = sm_sum["accept_SM"];
    
    beta_result.slice(t) = beta_final;
    arma::vec MHa = beta_sum["accept_MH"];
    MH_accept.row(t) = MHa.t();
    
    if(((t + 1) - (floor((t + 1)/thin) * thin)) == 0){
      atrisk_result.slice(save_col) = atrisk_init;
      ci_result.col(save_col) = ci_final;
      save_col += 1;
    }
    
    if(((t + 1) - (floor((t + 1)/500) * 500)) == 0){
      std::cout << "Iter: " << (t+1) << " - Done!" << std::endl;
    }
    
  }
  
  Rcpp::List result;
  result["ci_result"] = ci_result.t();
  result["beta_result"] = beta_result;
  result["atrisk_result"] = atrisk_result;
  result["sm_status"] = sm_status;
  result["sm_accept"] = sm_accept;
  result["MH_accept"] = MH_accept;
  return result;
  
}

// DM-x: --------------------------------------------------------------------

// Cluster Assignment: Reallocation
// [[Rcpp::export]]
arma::uvec realloc_dm(unsigned int Kmax, arma::mat z, arma::mat atrisk, 
                      arma::mat beta_mat, arma::uvec ci_old, double theta){
  
  arma::uvec ci_updated(ci_old);
  
  // Note that we allow the observation to go to all clusters.
  
  // Calculate the log marginal for every observation in each possible clusters.
  // We can do this as this step does not change beta vector.
  arma::mat lmar_realloc = logmar(z, atrisk, beta_mat);
  
  // std::cout << 1 << std::endl;
  
  // Find the number of the observations in each active cluster
  arma::vec nk(Kmax, arma::fill::zeros);
  for(int k = 0; k < Kmax; ++k){
    nk[k] += arma::accu(ci_old == k);
  } 
  
  // std::cout << 2 << std::endl;
  
  for(int i = 0; i < z.n_rows; ++i){
    int ci_old = ci_updated[i];
    
    // Adjust from Kmax space to Kpos space
    nk[ci_updated[i]] -= 1; 
    arma::vec log_realloc_prob = arma::log(nk + theta) + lmar_realloc.row(i).t();
    arma::vec realloc_prob = log_sum_exp(log_realloc_prob);
    arma::vec ck_new_vec = rmultinom_1(realloc_prob);
    arma::uvec k_new_vec = arma::find(ck_new_vec == 1);
    int k_new_ci = k_new_vec[0];
    nk[k_new_vec[0]] += 1;
    
    // Adjust from Kpos space back to Kmax space
    ci_updated[i] = k_new_ci;
    
  }
  
  return ci_updated;
  
}

// [[Rcpp::export]]
Rcpp::List DM_DM(unsigned int iter, unsigned int Kmax, 
                 arma::mat z, arma::mat beta_init, 
                 arma::uvec ci_init, double theta, double mu, double s2, 
                 double s2_MH, unsigned int t_thres,  
                 unsigned int thin){
  
  std::cout << "Total Iteration: " << iter << std::endl;
  std::cout << "Thinning: " << thin << std::endl;
  std::cout << "Total Recorded Iteration: " << iter/thin << std::endl;
  std::cout << "############ PERFORM THE CLUSTERING ############" << std::endl;
  unsigned int save_col = 0;
  
  arma::mat atrisk_mat(z.n_rows, z.n_cols, arma::fill::value(1.0));
  
  arma::umat ci_result(z.n_rows, iter/thin, arma::fill::value(Kmax + 1));
  
  arma::mat MH_accept(iter, Kmax, arma::fill::value(-2));
  arma::cube beta_result(Kmax, z.n_cols, iter, arma::fill::value(0));
  
  arma::mat beta0 = beta_init; 
  arma::mat beta_mcmc(beta_init); 
  Rcpp::List beta_sum; 
  arma::uvec ci_mcmc(ci_init);
  
  for(int t = 0; t < iter; ++t){

    // update beta
    beta_sum = update_beta_adaptive(t, t_thres, z, atrisk_mat, beta_result, beta0,
                                    ci_init, mu, s2, s2_MH);
    arma::mat bMCMC = beta_sum["beta_update"];
    // reallocation
    ci_mcmc = realloc_dm(Kmax, z, atrisk_mat, bMCMC, ci_init, theta);
    
    // record the result
    arma::mat beta_final = bMCMC;
    beta_init = bMCMC;
    
    arma::uvec ci_final = ci_mcmc;
    ci_init = ci_mcmc;
    
    beta_result.slice(t) = bMCMC;
    arma::vec MHa = beta_sum["accept_MH"];
    MH_accept.row(t) = MHa.t();
    
    if(((t + 1) - (floor((t + 1)/thin) * thin)) == 0){
      ci_result.col(save_col) = ci_final;
      save_col += 1;
    }
    
    if(((t + 1) - (floor((t + 1)/500) * 500)) == 0){
      std::cout << "Iter: " << (t+1) << " - Done!" << std::endl;
    }
    
  }
  
  Rcpp::List result;
  result["ci_result"] = ci_result.t();
  result["beta_result"] = beta_result;
  result["MH_accept"] = MH_accept;
  return result;
  
}

// DM-ZIDM: --------------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List DM_ZIDM(unsigned int iter, unsigned int Kmax, unsigned int nbeta_split, 
                   arma::mat z, arma::mat beta_init, 
                   arma::uvec ci_init, double theta, double mu, double s2, 
                   double s2_MH, unsigned int t_thres, unsigned int launch_iter, 
                   double r0c, double r1c, 
                   unsigned int thin){
  
  std::cout << "Total Iteration: " << iter << std::endl;
  std::cout << "Thinning: " << thin << std::endl;
  std::cout << "Total Recorded Iteration: " << iter/thin << std::endl;
  std::cout << "############ PERFORM THE CLUSTERING ############" << std::endl;
  unsigned int save_col = 0;
  
  arma::mat atrisk_mat(z.n_rows, z.n_cols, arma::fill::value(1.0));
  
  arma::umat ci_result(z.n_rows, iter/thin, arma::fill::value(Kmax + 1));
  arma::vec sm_status(iter, arma::fill::value(-2));
  arma::vec sm_accept(iter, arma::fill::value(-2));
  
  arma::mat MH_accept(iter, Kmax, arma::fill::value(-2));
  arma::cube beta_result(Kmax, z.n_cols, iter, arma::fill::value(0));
  
  arma::mat beta0 = beta_init; 
  arma::mat beta_mcmc(beta_init); 
  Rcpp::List sm_sum;
  Rcpp::List beta_sum; 
  arma::uvec ci_mcmc(ci_init);
  
  for(int t = 0; t < iter; ++t){
    // update beta
    beta_sum = update_beta_adaptive(t, t_thres, z, atrisk_mat, beta_result, beta0,
                                    ci_init, mu, s2, s2_MH);
    arma::mat bMCMC = beta_sum["beta_update"];
    // reallocation
    ci_mcmc = realloc_full(Kmax, z, atrisk_mat, bMCMC, ci_init, theta);
    // sm
    sm_sum = sm(Kmax, nbeta_split, z, atrisk_mat, bMCMC, ci_mcmc, 
                theta, mu, s2, launch_iter, r0c, r1c);
    
    // record the result
    arma::mat beta_final = sm_sum["beta_result"];
    beta_init = beta_final;
    
    arma::uvec ci_final = sm_sum["ci_result"];
    ci_init = ci_final;
    
    sm_status[t] = sm_sum["split_index"];
    sm_accept[t] = sm_sum["accept_SM"];
    
    beta_result.slice(t) = beta_final;
    arma::vec MHa = beta_sum["accept_MH"];
    MH_accept.row(t) = MHa.t();
    
    if(((t + 1) - (floor((t + 1)/thin) * thin)) == 0){
      ci_result.col(save_col) = ci_final;
      save_col += 1;
    }
    
    if(((t + 1) - (floor((t + 1)/500) * 500)) == 0){
      std::cout << "Iter: " << (t+1) << " - Done!" << std::endl;
    }
    
  }
  
  Rcpp::List result;
  result["ci_result"] = ci_result.t();
  result["beta_result"] = beta_result;
  result["sm_status"] = sm_status;
  result["sm_accept"] = sm_accept;
  result["MH_accept"] = MH_accept;
  return result;
  
}

// [[Rcpp::export]]
Rcpp::List ZIDM_DM(unsigned int iter, unsigned int Kmax, 
                 arma::mat z, arma::mat atrisk_init, arma::mat beta_init, 
                 arma::uvec ci_init, double theta, double mu, double s2, 
                 double s2_MH, unsigned int t_thres, double r0g, double r1g,
                 unsigned int thin){
  
  std::cout << "Total Iteration: " << iter << std::endl;
  std::cout << "Thinning: " << thin << std::endl;
  std::cout << "Total Recorded Iteration: " << iter/thin << std::endl;
  std::cout << "############ PERFORM THE CLUSTERING ############" << std::endl;
  unsigned int save_col = 0;
  
  arma::cube atrisk_result(z.n_rows, z.n_cols, iter/thin, arma::fill::value(0));
  arma::umat ci_result(z.n_rows, iter/thin, arma::fill::value(Kmax + 1));
  
  arma::mat MH_accept(iter, Kmax, arma::fill::value(-2));
  arma::cube beta_result(Kmax, z.n_cols, iter, arma::fill::value(0));
  
  arma::mat atrisk_mcmc(atrisk_init);
  arma::mat beta0 = beta_init; 
  arma::mat beta_mcmc(beta_init); 
  Rcpp::List beta_sum; 
  arma::uvec ci_mcmc(ci_init);
  
  for(int t = 0; t < iter; ++t){
    
    // update at-risk
    atrisk_mcmc = update_atrisk(z, atrisk_init, beta_init, ci_init, r0g, r1g);
    
    // update beta
    beta_sum = update_beta_adaptive(t, t_thres, z, atrisk_mcmc, beta_result, beta0,
                                    ci_init, mu, s2, s2_MH);
    arma::mat bMCMC = beta_sum["beta_update"];
    // reallocation
    ci_mcmc = realloc_dm(Kmax, z, atrisk_mcmc, bMCMC, ci_init, theta);
    
    // record the result
    atrisk_init = atrisk_mcmc;
    arma::mat beta_final = bMCMC;
    beta_init = bMCMC;
    
    arma::uvec ci_final = ci_mcmc;
    ci_init = ci_mcmc;
    
    beta_result.slice(t) = bMCMC;
    arma::vec MHa = beta_sum["accept_MH"];
    MH_accept.row(t) = MHa.t();
    
    if(((t + 1) - (floor((t + 1)/thin) * thin)) == 0){
      atrisk_result.slice(save_col) = atrisk_init;
      ci_result.col(save_col) = ci_final;
      save_col += 1;
    }
    
    if(((t + 1) - (floor((t + 1)/500) * 500)) == 0){
      std::cout << "Iter: " << (t+1) << " - Done!" << std::endl;
    }
    
  }
  
  Rcpp::List result;
  result["atrisk_result"] = atrisk_result;
  result["ci_result"] = ci_result.t();
  result["beta_result"] = beta_result;
  result["MH_accept"] = MH_accept;
  return result;
  
}

// *****************************************************************************