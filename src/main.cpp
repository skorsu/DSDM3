#include <RcppArmadillo.h>
//[[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;   

// Mathematical constants
#define MATH_PI        3.141592653589793238462643383279502884197169399375105820974
#define MATH_PI_2      1.570796326794896619231321691639751442098584699687552910487
#define MATH_2_PI      0.636619772367581343075535053490057448137838582961825794990
#define MATH_PI2       9.869604401089358618834490999876151135313699407240790626413
#define MATH_PI2_2     4.934802200544679309417245499938075567656849703620395313206
#define MATH_SQRT1_2   0.707106781186547524400844362104849039284835937688474036588
#define MATH_SQRT_PI_2 1.253314137315500251207882642405522626503493370304969158314
#define MATH_LOG_PI    1.144729885849400174143427351353058711647294812915311571513
#define MATH_LOG_2_PI  -0.45158270528945486472619522989488214357179467855505631739
#define MATH_LOG_PI_2  0.451582705289454864726195229894882143571794678555056317392

#define GETM(x, i, j)   x(i % x.nrow(), j)   // wrapped indexing of matrix

namespace helper {

  double lgamma_s( double x ){
    
    // Function for evaluating lgamma( x ) but with more numeric stability.
    
    double lg = -log( x );
    lg += lgamma( 1 + x );
    return lg;
    
  }

  double logmar_data( const arma::rowvec zi, 
                      const arma::rowvec at_risk_i, 
                      const arma::rowvec xi_k ) {
    
    // Define new variables
    arma::rowvec beta = exp( xi_k ); // Define beta = exp( xi )
    arma::rowvec ar_beta = at_risk_i % beta;
    
    double lm = 0.0;
    
    // Data part
    lm += lgamma( accu( zi ) + 1 );
    lm -= accu( lgamma( zi + 1 ) );
    
    // Dirichlet Part
    lm += lgamma( accu( ar_beta ) );
    lm -= accu( at_risk_i % lgamma( beta ) );
    
    // Dirichlet Integral Part
    lm += accu( at_risk_i % lgamma( beta + zi ) );
    lm -= lgamma( accu( ar_beta + zi ) );
    
    return lm;
    
  }

  double sum_loglik_k( const arma::mat z_k, 
                       const arma::mat at_risk_k, 
                       const arma::rowvec xi_k,
                       const unsigned int nk ){
    
    arma::mat betaM = repelem( exp( xi_k ), nk, 1 );
    arma::mat arbetaM = at_risk_k % betaM;
    arma::vec loglik( nk, fill::zeros );
    
    loglik += lgamma( sum( z_k, 1 ) + 1 );
    loglik -= sum( lgamma( z_k + 1 ), 1 );
    
    loglik += lgamma( sum( arbetaM, 1 ) );
    loglik -= sum( at_risk_k % lgamma( betaM ), 1 ); 
    
    loglik += sum( at_risk_k % lgamma( betaM + z_k ), 1 );
    loglik -= lgamma( sum( arbetaM + z_k, 1 ) );
    
    return accu( loglik );
    
  }

  arma::vec cpp_rdirichlet( const arma::vec& alpha ){
    
    // Citation Needed - https://github.com/twolodzko/extraDistr
    // Adjustment - Use the armadillo environment, fix n = 1
    
    int k = alpha.n_elem;
    arma::vec x( k, fill::zeros );
    double row_sum = 0.0;
    bool wrong_values = false;
    
    // Check if there are any columns in alpha
    if( k < 1 ) {
      
      warning( "NAs produced" );
      return arma::vec( k ).fill( NA_REAL );
      
    }
    
    if( k < 2 ){
      
      stop( "number of elements in alpha should be >= 2" );
      
    }
    
    for( int j = 0; j < k; ++j ){
      
      if( alpha( j ) <= 0 ){
        
        wrong_values = true;
        break;
        
      }
      
      x( j ) = R::rgamma( alpha( j ), 1.0 );
      row_sum += x( j );
      
    }
    
    if( row_sum == 0 || wrong_values ){
      
      warning( "NAs produced" );
      x.fill( NA_REAL );
      
    } else {
      
      x /= row_sum;
      
    }
    
    return x;
    
  }

  int sample_arma( const arma::vec& index,
                   const arma::vec& prob ) {
    
    int K = prob.n_elem; // Number of elements
    arma::vec cum_prob = arma::cumsum( prob / arma::sum( prob ) );
    double u = R::runif( 0, 1 ); // Generate a uniform random number between 0 and 1
    
    for( int i = 0; i < K; i++ ){
      
      if (u < cum_prob[i]) {
        
        return index[ i ];
        
      }
      
    }
    
    return index[ K - 1 ];
    
  }

  arma::vec LSE_prob( arma::vec lprob ){
    
    double c = max( lprob );
    double y = c + log( accu( exp( lprob - c ) ) );
    
    return exp( lprob - y );
    
  }

}

namespace sampler{

  void step1A( arma::vec& c,
               arma::vec& Nk,
               const arma::vec eta,
               const unsigned int K,
               const unsigned int N,
               const unsigned int J,
               const arma::mat z,
               const arma::mat at_risk,
               const arma::mat xi ){
    
    // This function updates c and cluster size
    
    arma::vec active_clus_list = regspace( 1, K );
    arma::vec log_p( K, fill::zeros );
    arma::vec p( K, fill::zeros );
    int c_new = 0;
    
    for( int i = 0; i < N; ++i ){
      
      Nk( c( i ) - 1 ) -= 1;
      log_p.zeros();
      p.zeros();
      
      for( int k = 0; k < K; ++k ){
        
        log_p( k ) += helper::logmar_data( z.row( i ), at_risk.row( i ), xi.row( k ) );
        log_p( k ) += log( eta( k ) );
        
      }

      p = helper::LSE_prob( log_p );
      
      c_new = helper::sample_arma( active_clus_list, p );
      
      Nk( c_new - 1 ) += 1;
      c( i ) = c_new;
      
    }
    
  }
  
  unsigned int step1B( arma::vec& c,
                       arma::mat& xi,
                       arma::vec& Nk,
                       arma::vec& eta,
                       const unsigned int N,
                       const unsigned int J,
                       const unsigned int Km ){
    
    // This function return the number of active clusters (Kp) and re-indexing c, xi, cluster size, component weight
    arma::uvec current_index = find( Nk != 0 ) + 1; 
    unsigned int Kp = current_index.size();
    
    arma::vec int_c( N, fill::zeros );
    arma::mat int_xi( Km, J, fill::zeros );
    arma::vec int_Nk( Km, fill::zeros );
    arma::vec int_eta( Km, fill::zeros );
    
    int old_clus = 0;
    int new_clus = 0;
    
    for( int k = 0; k < Kp; ++k ){
      
      old_clus = current_index( k );
      new_clus = k + 1;
      
      int_c.rows( find( c == old_clus ) ).fill( new_clus ); // Reindex c
      int_xi.row( k ) = xi.row( old_clus - 1 ); // Reindex xi
      int_Nk( k ) = Nk( old_clus - 1 ); // Reindex Nk
      int_eta( k ) = eta( old_clus - 1 ); // Reindex eta
      
    }
    
    c = int_c;
    xi = int_xi;
    Nk = int_Nk;
    eta = int_eta;
    
    return Kp;
    
  }
  
  void update_xi( arma::mat& xi,
                  const arma::vec c,
                  const unsigned int Kp,
                  const arma::vec Nk,
                  const arma::mat z,
                  const arma::mat at_risk,
                  const unsigned int J,
                  const arma::vec mean_J,
                  const double s2_prior,
                  const double s2_MH,
                  arma::vec& s2a_perform, // How many times that the model update xi_kj ( regardless the cluster and taxa )
                  arma::vec& s2a_accept ){ // How many time that we accept the proposed xi_kj ( regardless the cluster and taxa )
    
    // Update the cluster concentration (xi) parameters ( part of step 2A )
    
    arma::mat z_k;
    arma::mat ar_k;
    unsigned int n_k;
    double logA = 0.0;
    double logU = 0.0;
    arma::rowvec current_xi_k;
    arma::rowvec proposed_xi_k;
    double current_xi_kj = 0.0;
    double proposed_xi_kj = 0.0;
    
    for( int k = 0; k < Kp; ++k ){ // Loop through active cluster
      
      z_k = z.rows( find( c == ( k + 1 ) ) );
      ar_k = at_risk.rows( find( c == ( k + 1 ) ) );
      n_k = Nk( k );
      
      for( int j = 0; j < J; ++j ){ // Loop through taxa
        
        s2a_perform( 0 ) += 1;
        
        logA = 0.0;
        logU = log( randu( ) );
        
        current_xi_k = xi.row( k );
        current_xi_kj = current_xi_k( j );
        
        proposed_xi_kj = R::rnorm( current_xi_kj, sqrt( s2_MH ) );
        proposed_xi_k = current_xi_k;
        proposed_xi_k( j ) = proposed_xi_kj;
        
        logA += helper::sum_loglik_k( z_k, ar_k, proposed_xi_k, n_k );
        logA += log_normpdf( proposed_xi_kj, mean_J( j ), sqrt( s2_prior ) );
        logA -= helper::sum_loglik_k( z_k, ar_k, current_xi_k, n_k );
        logA -= log_normpdf( current_xi_kj, mean_J( j ), sqrt( s2_prior ) );
        
        if( logU <= logA ){
          
          s2a_accept( 0 ) += 1;
          xi( k, j ) = proposed_xi_kj;
          
        } 
        
      }
      
    } 
    
  }
  
  void update_at_risk( arma::mat& at_risk,
                       const arma::mat z, 
                       const arma::mat xi,
                       const arma::vec c,
                       const double a,
                       const double b,
                       const arma::mat zero_loc, // location of zij = 0
                       const unsigned int n_zero ){ // number of zero in z
    
    // Update the at risk indicator ( part of step 2A )
    
    unsigned int i;
    unsigned int j;
    arma::rowvec current_ar_i;
    arma::rowvec proposed_ar_i;
    arma::rowvec xi_k; 
    double logA = 0.0;
    double logU = 0.0;
    
    // Update the at-risk indicator for every zij = 0
    for( int i0 = 0; i0 < n_zero; ++i0 ){
      
      logA = 0.0;
      logU = log( randu(  ) );
      
      // Determine the location of zij = 0
      i = zero_loc( i0, 0 );
      j = zero_loc( i0, 1 );
      
      // Propose a new at-risk indicator (gamma_ij)
      current_ar_i = at_risk.row( i );
      proposed_ar_i = at_risk.row( i );
      proposed_ar_i( j ) = 1 - current_ar_i( j );
      
      // Obtain the xk for the current observation
      xi_k = xi.row( c( i ) - 1 );
      
      // Calculate the acceptance probability
      logA += helper::logmar_data( z.row( i ), proposed_ar_i, xi_k );
      logA += ( ( proposed_ar_i( j ) == 1 ) * a + ( proposed_ar_i( j ) == 0 ) * b );
      logA -= helper::logmar_data( z.row( i ), current_ar_i, xi_k );
      logA -= ( ( current_ar_i( j ) == 1 ) * a + ( current_ar_i( j ) == 0 ) * b );
      
      // Determine
      if( logU < logA ){
        
        at_risk( i, j ) = proposed_ar_i( j );
        
      }
      
    }
    
  }
  
  int step3A( const unsigned int Kp, 
              const unsigned int Km,
              const arma::vec log_pK,
              const double gm,
              const unsigned int N ){ 
    
    // This function returns an updated K.
    
    // Placeholder
    unsigned int n_Kp = Km - Kp + 1;
    arma::vec Kp_list = regspace( Kp, Km );
    arma::vec log_p( n_Kp, fill::zeros );
    int k = 0;
    double lp = 0.0;
    arma::vec p( n_Kp, fill::zeros );
    
    // Calculate log P( k|C, gamma )
    for( int ii = 0; ii < n_Kp; ++ii ){
      
      lp = 0.0;
      k = Kp_list( ii );
      
      lp += log_pK( k - 1 );
      lp += helper::lgamma_s( k + 1 );
      lp -= helper::lgamma_s( ( k - Kp ) + 1 );
      lp += helper::lgamma_s( gm * k );
      lp -= helper::lgamma_s( N + ( gm * k ) );
      
      log_p( ii ) = lp;
      
    }
    
    // Applying LSE
    p = helper::LSE_prob( log_p );

    // update K
    return helper::sample_arma( Kp_list, p );
    
  }
  
  double step3B( const double gm_current,
                 const arma::vec Nk,
                 const unsigned int N,
                 const unsigned int K,
                 const unsigned int Kp,
                 const double s2_3b,
                 const double nu_l,
                 const double nu_r,
                 const unsigned int iter,
                 arma::vec& accept_3B ){ 
    
    // This function returns an updated gamma.
    
    // Placeholder
    double logA = 0.0;
    double logU = log( randu() );
    
    double gm_proposed = exp( randn( distr_param( log( gm_current ), sqrt( s2_3b ) ) ) );
    
    // Calculate log of acceptance probability
    logA += R::df( gm_proposed, nu_l, nu_r, true );
    logA += helper::lgamma_s( gm_proposed * K );
    logA -= helper::lgamma_s( N + ( gm_proposed * K ) );
    
    logA -= R::df( gm_current, nu_l, nu_r, true );
    logA -= helper::lgamma_s( gm_current * K );
    logA += helper::lgamma_s( N + ( gm_current * K ) );
    
    logA += ( log( gm_current ) - log( gm_proposed ) ); // Proposal
    
    for( int k = 0; k < Kp; ++k ){
      
      logA += helper::lgamma_s( Nk( k ) + gm_proposed );
      logA -= helper::lgamma_s( gm_proposed );
      logA -= helper::lgamma_s( Nk( k ) + gm_current );
      logA += helper::lgamma_s( gm_current );
      
    }
    
    // Decision
    if( logU <= logA ){
      
      // Accept
      accept_3B( iter ) = 1;
      return gm_proposed;
      
    } else {
      
      // Reject
      accept_3B( iter ) = 0;
      return gm_current;
      
    } 
    
  }
  
  void step4( arma::mat& xi,
              arma::vec& eta,
              const double gm,
              const arma::vec Nk, 
              const unsigned int N,
              const unsigned int J,
              const unsigned int K,
              const unsigned int Kp,
              const unsigned int Km,
              const arma::vec mean_J,
              const double s2_prior ){ 
    
    // This function updates the xi and eta internally.
    
    // Placeholder
    unsigned int Kn = K - Kp;
    arma::vec eta_new( Km, fill::zeros );
    
    // Step 4a - Initialize xi for the empty clusters.
    // Perform only when K - Kp > 0
    if( Kn > 0 ){
      
      arma::uvec index_row = regspace< arma::uvec >( Kp + 1, K ) - 1;
      
      for( int j = 0; j < J; ++j ){
        
        xi( Kp, j, size( Kn, 1 ) ) = randn( Kn, distr_param( mean_J( j ), sqrt( s2_prior ) ) );
        
      }
      
    }
    
    // Step 4b - Update eta
    // If K = 1, return c( 1, rep( 0, Km - 1 ) ), otherwise, return c( D( K ), rep( 0, Km - K ) )

    if( K == 1 ){

      eta.zeros();
      eta( 0 ) = 1.0;

    } else {

      arma::vec e = Nk.head_rows( K ) + gm;
      eta.head_rows( K ) = helper::cpp_rdirichlet( e );

    }
    
  }

  void step4_MH( arma::mat& xi,
                 arma::vec& eta,
                 const double gm,
                 const arma::vec Nk, 
                 const unsigned int N,
                 const unsigned int J,
                 const unsigned int K,
                 const unsigned int Kp,
                 const unsigned int Km,
                 const arma::vec mu_prior,
                 const double s2_prior,
                 const double pi_J,
                 const arma::vec mu_MH_vec,
                 const double mu_MH,
                 const double s2_MH_1,
                 const double s2_MH_2,
                 arma::vec& s4a_perform, // How many times that the model update xi_kj ( regardless the cluster and taxa )
                 arma::vec& s4a_accept ){ // How many time that we accept the proposed xi_kj ( regardless the cluster and taxa )
    
    // This function updates the xi and eta internally.
    // Using the MH to proposed xi for the new clusters.
    
    // Placeholder
    unsigned int Kn = K - Kp;
    arma::vec eta_new( Km, fill::zeros );
    double current_xi;
    double proposed_xi; 
    double r_J = 0;
    double s2_MH = 0.0;
    double logA = 0.0;
    double logU = 0.0;
    
    // Step 4a - Initialize xi for the empty clusters.
    // Perform only when Kn > 0
    if( Kn > 0 ){
      
      for( int k = Kp; k < K; ++k ){
        
        for( int j = 0; j < J; ++j ){
          
          s4a_perform( 0 ) += 1;
          logA = 0.0;
          logU = log( randu( ) );
          
          current_xi = randn( distr_param( mu_prior( j ) , sqrt( s2_prior ) ) );
          r_J = ( randu( ) < pi_J );
          s2_MH = ( r_J * s2_MH_1 ) + ( ( 1 - r_J ) * s2_MH_2 );
          proposed_xi = randn( distr_param( mu_MH + mu_MH_vec( j ) , sqrt( s2_MH ) ) );
          
          logA += log_normpdf( proposed_xi, mu_prior( j ), sqrt( s2_prior ) );
          logA += log_normpdf( current_xi, mu_MH + mu_MH_vec( j ), sqrt( s2_MH ) );
          
          logA -= log_normpdf( current_xi, mu_prior( j ), sqrt( s2_prior ) );
          logA -= log_normpdf( proposed_xi, mu_MH + mu_MH_vec( j ), sqrt( s2_MH ) );
          
          if( logU < logA ){
            
            s4a_accept( 0 ) += 1;
            xi( k, j ) = proposed_xi;
            
          } else {
            
            xi( k, j ) = current_xi;
            
          }

        }
        
      }
      
    }
    
    // Step 4b - Update eta
    // If K = 1, return c( 1, rep( 0, Km - 1 ) ), otherwise, return c( D( K ), rep( 0, Km - K ) )
    
    if( K == 1 ){
      
      eta.zeros();
      eta( 0 ) = 1.0;
      
    } else {
      
      arma::vec e = Nk.head_rows( K ) + gm;
      eta.head_rows( K ) = helper::cpp_rdirichlet( e );
      
    }
    
  }

}

// [[Rcpp::export]]
arma::vec LSE_prob( arma::vec lprob ){
  
  double c = max( lprob );
  double y = c + log( accu( exp( lprob - c ) ) );
  
  return exp( lprob - y );
  
}

// [[Rcpp::export]]
arma::vec cpp_rdirichlet( const arma::vec& alpha ){
  
  // Citation Needed - https://github.com/twolodzko/extraDistr
  // Adjustment - Use the armadillo environment, fix n = 1
  
  int k = alpha.n_elem;
  arma::vec x( k, fill::zeros );
  double row_sum = 0.0;
  bool wrong_values = false;
  
  // Check if there are any columns in alpha
  if( k < 1 ) {
    
    warning( "NAs produced" );
    return arma::vec( k ).fill( NA_REAL );
    
  }
  
  if( k < 2 ){
    
    stop( "number of elements in alpha should be >= 2" );
    
  }
  
  for( int j = 0; j < k; ++j ){
    
    if( alpha( j ) <= 0 ){
      
      wrong_values = true;
      break;
      
    }
    
    x( j ) = R::rgamma( alpha( j ), 1.0 );
    row_sum += x( j );
    
  }
  
  if( row_sum == 0 || wrong_values ){
    
    warning( "NAs produced" );
    x.fill( NA_REAL );
    
  } else {
    
    x /= row_sum;
    
  }
  
  return x;
  
}

// [[Rcpp::export]]
double logmar_data( const arma::rowvec zi, 
                    const arma::rowvec at_risk_i, 
                    const arma::rowvec xi_k ) {
  
  // Define new variables
  arma::rowvec beta = exp( xi_k ); // Define beta = exp( xi )
  arma::rowvec ar_beta = at_risk_i % beta;
  
  double lm = 0.0;
  
  // Data part
  lm += lgamma( accu( zi ) + 1 );
  lm -= accu( lgamma( zi + 1 ) );
  
  // Dirichlet Part
  lm += lgamma( accu( ar_beta ) );
  lm -= accu( at_risk_i % lgamma( beta ) );
  
  // Dirichlet Integral Part
  lm += accu( at_risk_i % lgamma( beta + zi ) );
  lm -= lgamma( accu( ar_beta + zi ) );
  
  return lm;
  
}

// [[Rcpp::export]]
double log_pPoisson( int k, double lambda ) {
  
  double log_p = 0.0;
  log_p += ( -lambda );
  log_p += k * log( lambda );
  log_p -= helper::lgamma_s( k + 1 ); 
  log_p -= log( 1 - exp( -lambda ) );  // Normalization for truncation
  return log_p;
  
}

// [[Rcpp::export]]
double log_pTB( double k, double Km, double pi ){
  
  // Calculate the P( X = k ) in a log scale when X ~ Zero-Truncated-Binomial( Km, pi )
  
  double logp = 0.0;
  
  logp += helper::lgamma_s( Km + 1 );
  logp -= helper::lgamma_s( ( Km - k ) + 1 );
  logp -= helper::lgamma_s( k + 1 );
  
  logp += ( k * log( pi ) );
  logp += ( ( Km - k ) * log( 1 - pi ) );
  
  logp -= log( 1 - pow( 1 - pi, Km ) );
  
  return logp;
  
}

// void mod_fixed_MH( arma::vec& c_init, // Initialization
//                    arma::mat& xi_init, // Initialization
//                    const arma::mat z,
//                    const arma::mat zero_loc,
//                    const unsigned int Km,
//                    const arma::vec log_pK,
//                    const arma::vec mu_prior,
//                    const double s2_prior,
//                    const double a_ar,
//                    const double b_ar,
//                    const double s2_MH_xi,
//                    const double gm,
//                    const double pi_J,
//                    const arma::vec mu_MH_vec,
//                    const double mu_MH,
//                    const double s2_MH_1,
//                    const double s2_MH_2,
//                    const unsigned int iter,
//                    const double thin,
//                    arma::mat& c_mat, // Store the result
//                    arma::vec& Kp_vec, // Store the result
//                    arma::vec& K_vec, // Store the result
//                    arma::mat& eta_mat, // Store the result
//                    arma::cube& xi_cube, // Store the result
//                    arma::vec& s2a_perform, // Step 2A: How many times that the model update xi_kj ( regardless the cluster and taxa )
//                    arma::vec& s2a_accept, // Step 2A: How many time that we accept the proposed xi_kj ( regardless the cluster and taxa )
//                    arma::vec& s4a_perform, // Step 4A: How many times that the model update xi_kj ( regardless the cluster and taxa )
//                    arma::vec& s4a_accept ){ // Step 4A: How many time that we accept the proposed xi_kj ( regardless the cluster and taxa )
//   
//   unsigned int N = z.n_rows;
//   unsigned int J = z.n_cols;
//   unsigned int Kp = 0;
//   unsigned int K = 0;
//   int n0 = zero_loc.n_rows;
//   int save_xi = 0;
//   
//   // Obtain Nk and Kp from c_init
//   arma::vec Nk( Km, fill::zeros );
//   for( int k = 0; k < Km; ++k ){
//     
//     Nk( k ) += accu( c_init == ( k + 1 ) );
//     Kp += ( Nk( k ) != 0 );
//     
//   }
//   
//   // std::cout << "Init Nk: " << Nk.t() << std::endl;
//   
//   K = Kp;
//   
//   arma::mat ar_init( N, J, fill::ones );
//   arma::vec eta_init( Km, fill::zeros );
//   eta_init.head( K ).fill( 1 / static_cast< double >( K ) );
//   
//   // std::cout << "Init eta: " << eta_init.t() << std::endl;
//   
//   for( int it = 0; it < iter; ++it ){
//     
//     // std::cout << "Kp ( Before Step 1 ): " << Kp << std::endl;
//     
//     // Step 1
//     if( K > 1 ){
//       sampler::step1A( c_init, Nk, eta_init, K, N, J, z, ar_init, xi_init );
//     } 
//     
//     Kp = sampler::step1B( c_init, xi_init, Nk, eta_init, N, J, Km );
//     
//     // Step 2A
//     sampler::update_xi( xi_init, c_init, Kp, Nk, z, ar_init, J, mu_prior, s2_prior, s2_MH_xi, s2a_perform, s2a_accept );
//     if( n0 > 0 ){
//       sampler::update_at_risk( ar_init, z, xi_init, c_init, a_ar, b_ar, zero_loc, n0 );
//     }
// 
//     // Step 3
//     K = sampler::step3A( Kp, Km, log_pK, gm, N );
//     
//     // Step 4
//     sampler::step4_MH( xi_init, eta_init, gm, Nk, N, J, K, Kp, Km, mu_prior, s2_prior, pi_J, mu_MH_vec, mu_MH, s2_MH_1, s2_MH_2, s4a_perform, s4a_accept );
//     
//     // Record the result
//     Kp_vec( it ) = Kp;
//     K_vec( it ) = K;
//     c_mat.row( it ) = c_init.t();
//     eta_mat.row( it ) = eta_init.t();
//     
//     if( ( ( it + 1 ) - floor( ( it + 1 ) / thin ) * thin ) == 0 ){
//       
//       xi_cube.slice( save_xi ) = xi_init;
//       save_xi += 1;
//       
//     } 
//     
//   } 
//   
// } 

// [[Rcpp::export]]
void mod_fixed( arma::vec& c_init, // Initialization
                arma::mat& xi_init, // Initialization
                const arma::mat z,
                const arma::mat zero_loc,
                const unsigned int Km,
                const arma::vec log_pK,
                const arma::vec Mean_vec,
                const double s2_prior,
                const double a_ar,
                const double b_ar,
                const double s2_MH_xi,
                const double gm,
                const unsigned int iter,
                const double thin,
                arma::mat& c_mat, // Store the result
                arma::vec& Kp_vec, // Store the result
                arma::vec& K_vec, // Store the result
                arma::mat& eta_mat, // Store the result
                arma::cube& xi_cube,
                arma::vec& s2a_perform, // Step 2A: How many times that the model update xi_kj ( regardless the cluster and taxa )
                arma::vec& s2a_accept ){ // Store the result

  unsigned int N = z.n_rows;
  unsigned int J = z.n_cols;
  unsigned int Kp = 0;
  unsigned int K = 0;
  int n0 = zero_loc.n_rows;
  int save_xi = 0;

  // Obtain Nk and Kp from c_init
  arma::vec Nk( Km, fill::zeros );
  for( int k = 0; k < Km; ++k ){

    Nk( k ) += accu( c_init == ( k + 1 ) );
    Kp += ( Nk( k ) != 0 );

  }

  K = Kp;
  
  arma::mat ar_init( N, J, fill::ones );
  arma::vec eta_init( Km, fill::zeros );
  eta_init.head( K ).fill( 1 / static_cast< double >( K ) );

  for( int it = 0; it < iter; ++it ){

    // Step 1
    if( K > 1 ){
      sampler::step1A( c_init, Nk, eta_init, K, N, J, z, ar_init, xi_init );
    }

    Kp = sampler::step1B( c_init, xi_init, Nk, eta_init, N, J, Km );

    // Step 2A
    sampler::update_xi( xi_init, c_init, Kp, Nk, z, ar_init, J, Mean_vec, s2_prior, s2_MH_xi, s2a_perform, s2a_accept );
    if( n0 > 0 ){
      sampler::update_at_risk( ar_init, z, xi_init, c_init, a_ar, b_ar, zero_loc, n0 );
    }

    // Step 3
    K = sampler::step3A( Kp, Km, log_pK, gm, N );

    // Step 4
    sampler::step4( xi_init, eta_init, gm, Nk, N, J, K, Kp, Km, Mean_vec, s2_prior );

    // Record the result
    Kp_vec( it ) = Kp;
    K_vec( it ) = K;
    c_mat.row( it ) = c_init.t();
    eta_mat.row( it ) = eta_init.t();

    if( ( ( it + 1 ) - floor( ( it + 1 ) / thin ) * thin ) == 0 ){

      xi_cube.slice( save_xi ) = xi_init;
      save_xi += 1;

    }

  }

}

// void mod( arma::vec& c_init, // Initialization
//           arma::mat& xi_init, // Initialization
//           const arma::mat z,
//           const arma::mat zero_loc,
//           const unsigned int Km,
//           const arma::vec log_pK,
//           const arma::vec Mean_vec,
//           const double s2_prior,
//           const double a_ar,
//           const double b_ar,
//           const double nu_l,
//           const double nu_r,
//           const double s2_MH_xi,
//           const double s2_MH_gm, 
//           const unsigned int iter,
//           const double thin,
//           arma::mat& c_mat, // Store the result
//           arma::vec& accept_3B, // Store the result
//           arma::vec& Kp_vec, // Store the result
//           arma::vec& K_vec, // Store the result
//           arma::vec& gm_vec, // Store the result
//           arma::mat& eta_mat, // Store the result
//           arma::cube& xi_cube ){ // Store the result
//   
//   unsigned int N = z.n_rows;
//   unsigned int J = z.n_cols;
//   unsigned int Kp = 0;
//   unsigned int K = 0;
//   int n0 = zero_loc.n_rows;
//   double gm_init = R::rf( nu_l, nu_r );
//   int save_xi = 0;
//   
//   // Obtain Nk and Kp from c_init
//   arma::vec Nk( Km, fill::zeros );
//   for( int k = 0; k < Km; ++k ){
//     
//     Nk( k ) += accu( c_init == ( k + 1 ) );
//     Kp += ( Nk( k ) != 0 );
//     
//   }
//   
//   K = ( ( ( Kp + 1 ) < Km ) * ( Kp + 1 ) ) + ( ( ( Kp + 1 ) >= Km ) * ( Km ) );
//   
//   arma::mat ar_init( N, J, fill::ones );
//   arma::vec eta_init( Km, fill::zeros );
//   eta_init.head( K ).fill( 1 / static_cast< double >( K ) );
//   
//   for( int it = 0; it < iter; ++it ){
//     
//     // Step 1
//     if( K > 1 ){
//       sampler::step1A( c_init, Nk, eta_init, K, N, J, z, ar_init, xi_init );
//     }
//     
//     Kp = sampler::step1B( c_init, xi_init, Nk, eta_init, N, J, Km );
//     
//     // Step 2A
//     sampler::update_xi( xi_init, c_init, Kp, Nk, z, ar_init, J, Mean_vec, s2_prior, s2_MH_xi );
//     if( n0 > 0 ){
//       sampler::update_at_risk( ar_init, z, xi_init, c_init, a_ar, b_ar, zero_loc, n0 );
//     }
//     
//     // Step 3
//     K = sampler::step3A( Kp, Km, log_pK, gm_init, N );
//     gm_init = sampler::step3B( gm_init, Nk, N, K, Kp, s2_MH_gm, nu_l, nu_r, it, accept_3B );
//     
//     // Step 4
//     sampler::step4( xi_init, eta_init, gm_init, Nk, N, J, K, Kp, Km, Mean_vec, s2_prior );
//     
//     // Record the result
//     Kp_vec( it ) = Kp;
//     K_vec( it ) = K;
//     gm_vec( it ) = gm_init;
//     c_mat.row( it ) = c_init.t();
//     eta_mat.row( it ) = eta_init.t();
//     
//     if( ( ( it + 1 ) - floor( ( it + 1 ) / thin ) * thin ) == 0 ){
//       
//       xi_cube.slice( save_xi ) = xi_init;
//       save_xi += 1;
//       
//     }
//     
//   }
//   
// }
