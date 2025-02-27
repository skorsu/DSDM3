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

// Define helper functions
namespace helper{

  double log_pTB( double k, double Km, double pi ){
    
    // Calculate the P( X = k ) in a log scale when X ~ Zero-Truncated-Binomial( Km, pi )
    
    double logp = 0.0;
    
    logp += lgamma( Km + 1 );
    logp -= lgamma( ( Km - k ) + 1 );
    logp -= lgamma( k + 1 );
    
    logp += ( k * log( pi ) );
    logp += ( ( Km - k ) * log( 1 - pi ) );
    
    logp -= log( 1 - pow( 1 - pi, Km ) );
    
    return logp;
    
  }

  arma::mat mvrnormArma( int n, arma::vec mu, arma::mat sigma ) {
    int ncols = sigma.n_cols;
    arma::mat Y = arma::randn( n, ncols );
    return arma::repmat( mu, 1, n ).t()  + Y * arma::chol( sigma );
  }

  double ldnormARMA( arma::rowvec x, double mu, double s2, double J ) {
    double dd = accu( pow( x - mu, 2.0 ) );
    dd *= -pow( 2 * s2, -1 );
    dd -= ( ( J / 2 ) * log( 2 * MATH_PI * s2 ) );
    return dd;
  }

  double logmar_data( const arma::rowvec zi, 
                      const arma::rowvec at_risk_i, 
                      const arma::rowvec xi_k ) {
    arma::rowvec beta = exp( xi_k ); // Define beta = exp( xi )
    
    arma::rowvec ar_beta = at_risk_i % beta;
    double lm = 0.0;
    lm += lgamma( accu( zi ) + 1 );
    lm -= accu( lgamma( zi + 1 ) );
    lm += lgamma( accu( ar_beta ) );
    lm -= accu( at_risk_i % lgamma( beta ) );
    lm += accu( at_risk_i % lgamma( beta + zi ) );
    lm -= lgamma( accu( ar_beta + zi ) );
    return lm;
  }
  
  double lse( arma::vec x ){
    // x is the vector of the value in a log-scale.
    double xm = max( x );
    return xm + log( accu( exp( x - xm ) ) );
  }
  
  arma::vec norm_prob( arma::vec lprob ){
    
    arma::vec prob_adjust;
    
    double y = lse( lprob );
    arma::vec prob = exp( lprob - y );
    prob.clean( pow( 10, -20 ) );
    
    prob_adjust = normalise( prob, 1 );
    prob_adjust.replace( 0.0, pow( 10, -20 ) );
    
    return prob_adjust;
    
  }
  
  arma::uvec myseq( int first, int last) {
    arma::uvec y(abs(last - first) + 1);
    if (first < last)
      std::iota(y.begin(), y.end(), first);
    else {
      std::iota(y.begin(), y.end(), last);
      std::reverse(y.begin(), y.end());
    }
    return y;
  } 
  
  double sample_prob_cpp( IntegerVector x, NumericVector prob ){
    Function f("sample");
    IntegerVector sampled = f(x, Named("size") = 1, Named("prob") = prob);
    return sampled[0];
  }
  
  arma::vec convertIntegerVectorToArmaVec( IntegerVector x ) {
    arma::vec y(x.size());
    for(int i = 0; i < x.size(); i++) {
      y[i] = static_cast<double>(x[i]);
    }
    return y;
  }
  
  void mean_update( arma::mat& xbar_previous,
                    arma::vec& t,
                    const arma::mat data_current,
                    const arma::vec c,
                    unsigned int Km ){
    // Find the active clusters
    arma::vec active_clus = unique( c );
    unsigned int Kp = active_clus.size();
    // Update mean
    for( int kk = 0; kk < Kp; ++kk ){
      xbar_previous.row( active_clus( kk ) - 1 ) *= ( t( active_clus( kk ) - 1 ) - 1 );
      xbar_previous.row( active_clus( kk ) - 1 ) += data_current.row( active_clus( kk ) - 1 );
      xbar_previous.row( active_clus( kk ) - 1 ) /= t( active_clus( kk ) - 1 );
      t( active_clus( kk ) - 1 ) += 1;
    }
    // Reset t
    arma::vec ni( Km, arma::fill::zeros );
    for( int i = 0; i < c.size(); ++i ){
      ni( c( i ) - 1 ) += 1;
    }
    arma::uvec emp = arma::find( ni == 0 );
    for( int kk = 0; kk < emp.size(); ++kk ){
      t( emp ).fill( 1 );
    }
  }
  
  void cov_update( arma::cube& cov_previous,
                   const arma::mat xbar_previous,
                   const arma::vec t,
                   const arma::mat data_current,
                   const arma::vec c ){
    arma::mat dev_current = data_current - xbar_previous;
    // Find the active clusters
    arma::vec active_clus = unique( c );
    unsigned int Kp = active_clus.size();
    // Update covariance matrix
    for( int kk = 0; kk < Kp; ++kk ){
      if( t( active_clus( kk ) - 1 ) == 1 ){
        cov_previous.slice( active_clus( kk ) - 1 ).fill( 0 );
      } else {
        cov_previous.slice( active_clus( kk ) - 1 ) *= ( t( active_clus( kk ) - 1 ) - 2 );
        cov_previous.slice( active_clus( kk ) - 1 ) /= ( t( active_clus( kk ) - 1 ) - 1 );
        cov_previous.slice( active_clus( kk ) - 1 ) += ( ( 1 / t( active_clus( kk ) - 1 ) ) * dev_current.row( ( active_clus( kk ) - 1 ) ).t() * dev_current.row( active_clus( kk ) - 1 ) );
      }
    }
  }
  
  double log_probGS_c( const arma::vec c,
                       const arma::vec c_star,
                       arma::vec nc_star,// The cluster size
                       const arma::mat xi,
                       const arma::mat z,
                       const arma::mat at_risk,
                       const arma::uvec S, // The set S defined in Split-Merge paper 
                       const arma::vec clus_sm ){ // Two clusters that involve in the Spilt-merge process ( index from 1 to Km )
    
    // Function for calculating P_GS( c| c_star, xi, y ) in a log scale
    
    // Placeholder
    double lprob_gs = 0.0;
    arma::vec lprob( 2, arma::fill::zeros );
    arma::vec prob( 2, arma::fill::zeros );
    int clus0 = clus_sm( 0 ) - 1;
    int clus1 = clus_sm( 1 ) - 1;
    unsigned int nS = S.size();
    int i = -1;
    int ci = -1;
    int ci_star = -1;
    
    // Restricted Gibbs Scan for cluster assignment ( ci )
    for( int s = 0; s < nS; ++s ){
      
      lprob.zeros(); // Reset the lprob vector
      
      i = S[ s ]; // Obtain the index
      ci_star = c_star( i );
      nc_star( ci_star - 1 ) -= 1; // Remove ci from nc
      
      // Calculate the log probability
      lprob( 0 ) += log( nc_star( clus0 ) );
      lprob( 0 ) += helper::logmar_data( z.row( i ), at_risk.row( i ), xi.row( clus0 ) );
      lprob( 1 ) += log( nc_star( clus1 ) );
      lprob( 1 ) += helper::logmar_data( z.row( i ), at_risk.row( i ), xi.row( clus1 ) );
      
      prob = helper::norm_prob( lprob ); // Using log-sum-exp, obtain probability
      
      // std::cout << "prob: " << prob.t() << std::endl;
      
      nc_star( ci_star - 1 ) += 1; // Put it back
      
      // Calculate the log of P_GS( c_i| c_star_i, xi_star, y )
      ci = c( i );
      lprob_gs += log( ( ( ( ci - 1 ) == clus0 ) * prob( 0 ) ) + ( ( ( ci - 1 ) == clus1 ) * prob( 1 ) ) );
      
      // std::cout << "lprob_gs: " << lprob_gs << std::endl;
      
      
    }
    
    return lprob_gs;
    
  }

  void update_xi( arma::mat& current_xi,
                  const arma::vec c,
                  const arma::mat z,
                  const arma::mat at_risk,
                  const double mu_prior, // Prior Parameter - Use for evaluating
                  const double s2_prior, // Prior Parameter - Use for evaluating
                  const double s2_MH, // MH Parameter - Use for proposing
                  const arma::vec clus_int, // Clusters we want to update xi ( index from 1 to Km )
                  const double J ){
    
    // Placeholder
    unsigned int Kp = clus_int.size();
    double logU = 0.0;
    double logA = 0.0;
    arma::rowvec current_xi_k( J, arma::fill::zeros );
    arma::rowvec proposed_xi_k( J, arma::fill::zeros );
    arma::mat z_k; // Store the data for the interested cluster
    arma::mat ar_k; // Store the at-risk indicator for the interested cluster
    int k = 0;
    
    // Create the covariance matrix for MH. 
    arma::mat s2_mat( J, J, fill::eye );
    s2_mat *= s2_MH;
    
    // Update cluster concentration (xi) for each active cluster
    for( int kk = 0; kk < Kp; ++kk ){
      
      k = clus_int( kk ) - 1; // Find the index of the current cluster
      z_k = z.rows( find( c == ( k + 1 ) ) );
      ar_k = at_risk.rows( find( c == ( k + 1 ) ) );
      
      logA = 0.0; // Reset logA
      logU = log( randu( 1 )[ 0 ] );
      
      // Propose a new cluster concentration parameters
      current_xi_k = current_xi.row( k );
      proposed_xi_k = helper::mvrnormArma( 1, current_xi_k.t(), s2_mat ).row( 0 ); 
      
      // Calculate acceptance probability
      logA += helper::ldnormARMA( proposed_xi_k, mu_prior, s2_prior, J );
      logA -= helper::ldnormARMA( current_xi_k, mu_prior, s2_prior, J );
      
      for( int i = 0; i < z_k.n_rows; ++i ){
        logA += helper::logmar_data( z_k.row( i ), ar_k.row( i ), proposed_xi_k );
        logA -= helper::logmar_data( z_k.row( i ), ar_k.row( i ), current_xi_k );
      }
      
      // Determine
      if( logU < logA ){
        current_xi.row( k ) = proposed_xi_k;
      }
      
    }
    
  }

  void rGibbs( arma::vec& current_c,
               arma::mat& current_xi,
               arma::vec& nc, // The cluster size
               const arma::mat z,
               const arma::mat at_risk,
               const double mu_prior, // Prior Parameter - Use for evaluating
               const double s2_prior, // Prior Parameter - Use for evaluating
               const double s2_MH, // MH Parameter - Use for proposing
               const arma::uvec S, // The set S defined in Split-Merge paper 
               const arma::vec clus_sm, // Two clusters that involve in the Spilt-merge process ( index from 1 to Km )
               const double J ){ 
    
    // Placeholder
    arma::vec lprob( 2, arma::fill::zeros );
    arma::vec prob( 2, arma::fill::zeros );
    int clus0 = clus_sm( 0 ) - 1;
    int clus1 = clus_sm( 1 ) - 1;
    unsigned int nS = S.size();
    int i = -1;
    int ci = -1;
    double u = 0.0;
    
    // Restricted Gibbs Scan for cluster assignment ( ci )
    for( int s = 0; s < nS; ++s ){
      
      lprob.zeros(); // Reset the lprob vector
      
      i = S[ s ]; // Obtain the index
      ci = current_c( i );
      
      nc( ci - 1 ) -= 1; // Remove ci from nc
      
      // Calculate the log probability
      lprob( 0 ) += log( nc( clus0 ) );
      lprob( 0 ) += helper::logmar_data( z.row( i ), at_risk.row( i ), current_xi.row( clus0 ) );
      lprob( 1 ) += log( nc( clus1 ) );
      lprob( 1 ) += helper::logmar_data( z.row( i ), at_risk.row( i ), current_xi.row( clus1 ) );
      
      prob = helper::norm_prob( lprob ); // Using log-sum-exp, obtain probability
      
      // Determine and update ci
      u = randu();
      ci = ( ( clus0 + 1 ) * ( u < prob( 0 ) ) ) + ( ( clus1 + 1 ) * ( u >= prob( 0 ) ) );
      current_c( i ) = ci;
      nc( ci - 1 ) += 1;
      
    }
    
    // Restricted Gibbs update for cluster concentration ( xi )
    helper::update_xi( current_xi, current_c, z, at_risk, mu_prior, s2_prior, s2_MH, clus_sm, J );
    
  }

  double log_probGS_xi_k( const arma::rowvec xi_k,
                          const arma::vec c_star,
                          const arma::mat z,
                          const arma::mat at_risk,
                          const double mu_prior, // Prior Parameter - Use for evaluating
                          const double s2_prior, // Prior Parameter - Use for evaluating
                          const int k, // Cluster that we want to calculate P_GS ( index from 1 to Km )
                          const double J ){ 
    
    // Function for calculating P_GS( xi_k| xi_k_star, c_star, y ) in a log scale
    // This term reduced to P( xi_k | c_star, y ) which is proportional to P( y | xi_k, c_star = k ) * p( xi_k )
    
    // Placeholder
    double lprob_gs = helper::ldnormARMA( xi_k, mu_prior, s2_prior, J );
    arma::uvec ck = find( c_star == k );
    int i = 0;
    
    for( int ii = 0; ii < ck.size(); ++ii ){
      
      i = ck( ii );
      lprob_gs += helper::logmar_data( z.row( i ), at_risk.row( i ), xi_k );
      
    }
    
    return lprob_gs;
    
  }

}

// Define sampler functions
namespace sampler{
  
  void update_at_risk( arma::mat& current_ar,
                       const arma::mat z, 
                       const arma::mat xi,
                       const arma::vec c,
                       const double a,
                       const double b,
                       const arma::mat zero_loc, // location of zij = 0
                       const unsigned int n_zero ){ // number of zero in z
    
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
      logU = log( randu( 1 )[ 0 ] );
      // Determine the location of zij = 0
      i = zero_loc( i0, 0 ) - 1;
      j = zero_loc( i0, 1 ) - 1;
      // Propose a new at-risk indicator (gamma_ij)
      current_ar_i = current_ar.row( i );
      proposed_ar_i = current_ar.row( i );
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
        current_ar( i, j ) = proposed_ar_i( j );
      }
    }
    
  }
  
  
  void update_xi_ADAP( arma::mat& current_xi,
                       const arma::mat z,
                       const arma::vec c,
                       const arma::mat at_risk,
                       const double mu_prior,
                       const double s2_prior,
                       const double J,
                       const arma::cube cov_current,
                       const arma::vec t_vec,
                       const int t_thres ){
    arma::vec active_index = unique( c );
    unsigned int Kp = active_index.size();
    double sd = pow( 2.4, 2 )/J;
    double eps = pow( 10, -10 ); 
    arma::mat Ct( J, J, fill::zeros );
    arma::mat C0( J, J, fill::eye );
    arma::mat IJ( J, J, arma::fill::eye );
    C0 *= pow( 10, -3 );
    IJ *= eps;
    double logU = 0.0;
    double logA = 0.0;
    arma::rowvec current_xi_k;
    arma::rowvec proposed_xi_k;
    arma::mat z_k;
    arma::mat ar_k;
    for( int kk = 0; kk < Kp; ++kk ){
      logA = 0.0;
      logU = log( randu( 1 )[ 0 ] );
      current_xi_k = current_xi.row( active_index( kk ) - 1 );
      Ct = ( ( t_vec( active_index( kk ) - 1 ) <= t_thres ) * C0 ) + ( ( t_vec( active_index( kk ) - 1 ) > t_thres ) * sd * ( IJ + cov_current.slice( active_index( kk ) - 1 ) ) );
      proposed_xi_k = helper::mvrnormArma( 1, current_xi_k.t(), Ct ).row( 0 ); // Propose new xi
      z_k = z.rows( find( c == active_index( kk ) ) );
      ar_k = at_risk.rows( find( c == active_index( kk ) ) );
      // Calculate acceptance probability
      logA += helper::ldnormARMA( proposed_xi_k, mu_prior, s2_prior, J );
      logA -= helper::ldnormARMA( current_xi_k, mu_prior, s2_prior, J );
      for( int i = 0; i < z_k.n_rows; ++i ){
        logA += helper::logmar_data( z_k.row( i ), ar_k.row( i ), proposed_xi_k );
        logA -= helper::logmar_data( z_k.row( i ), ar_k.row( i ), current_xi_k );
      }
      // Determine
      if( logU < logA ){
        current_xi.row( active_index( kk ) - 1 ) = proposed_xi_k;
      }
    }
  }

  void alg8( arma::vec& current_c,
             arma::mat& current_xi,
             arma::vec& nc, // The cluster size
             unsigned int Km,
             const arma::mat z,
             const arma::mat at_risk,
             const arma::vec log_Vn,
             const unsigned int n,
             const unsigned int J,
             const double theta,
             const double mu_prior, // Prior Parameter
             const double s2_prior, // Prior Parameter
             const double s2_MH ){
    
    // Placeholder
    arma::uvec active_index = find( nc != 0 ) + 1; // The active cluster ( index from 1 to Km ) -- need to update in the loop
    arma::uvec inactive_index = find( nc == 0 ) + 1; // The inactive cluster ( index from 1 to Km ) -- need to update in the loop
    unsigned int Kp = active_index.size();
    unsigned int Kp_new = Kp;
    int ci = 0;
    unsigned int proposed_cluster = 0;
    arma::vec mu_prior_vec( J, fill::value( mu_prior ) );
    arma::mat s2_prior_mat( J, J, fill::eye );
    s2_prior_mat *= s2_prior;
    arma::uvec append_clus( 1 ); 
    arma::uvec scan_index;
    int k_update = 0;
    arma::vec prob_scan;
    IntegerVector hold;
    int c_index_f;
    
    // Reallocation
    for( int i = 0; i < n; ++i ){
      
      ci = current_c( i );
      nc( ci - 1 ) -= 1;
      
      if( nc( ci - 1 ) == 0 ){
        
        // Use the same index to be a new cluster.
        Kp_new = Kp;
        Kp -= 1;
        proposed_cluster = ci;
        current_xi.row( proposed_cluster - 1 ) = helper::mvrnormArma( 1, mu_prior_vec, s2_prior_mat ).row( 0 );
        append_clus( 0 ) = proposed_cluster;
        
      } else {
        
        if( Kp < Km ){
          
          // Propose a new cluster with the new index (given that we still have some available)
          Kp_new = Kp + 1;
          proposed_cluster = inactive_index( 0 );
          current_xi.row( proposed_cluster - 1 ) = helper::mvrnormArma( 1, mu_prior_vec, s2_prior_mat ).row( 0 );
          append_clus( 0 ) = proposed_cluster;
          
        } else {
          
          Kp_new = Kp;
          
        }
        
      }
      
      // Calculate the log allocation probability
      arma::vec lprob( Kp_new, fill::zeros ); 
      if( Kp != Kp_new ){
        
        scan_index = unique( join_cols( active_index, append_clus ) );
        lprob( find( append_clus( 0 ) == scan_index ) ) += log_Vn( Kp_new - 1 );
        lprob( find( append_clus( 0 ) == scan_index ) ) -= log_Vn( Kp - 1 );
        
      } else {
        
        scan_index = active_index;
        
      }
      
      for( int kk = 0; kk < Kp_new; ++kk ){
        
        k_update = scan_index( kk );
        lprob( kk ) += helper::logmar_data( z.row( i ), at_risk.row( i ), current_xi.row( k_update - 1 ) );
        
        if( Kp != Kp_new and k_update != append_clus( 0 ) ){
          
          lprob( kk ) += log( nc( k_update - 1 ) + theta );
          
        }
        
      }
      
      // Assign a cluster ot the current observation
      prob_scan = helper::norm_prob( lprob );
      hold = helper::myseq( 0, Kp_new - 1 );
      c_index_f = helper::sample_prob_cpp( hold, wrap( prob_scan ) );
      current_c( i ) = scan_index( c_index_f );
      nc( current_c( i ) - 1 ) += 1;
      
      // Adjust Kp
      active_index = find( nc != 0 ) + 1;
      inactive_index = find( nc == 0 ) + 1;
      Kp = active_index.size();
      
    }
    
    arma::uvec clus_int_uvec = find( nc != 0 ) + 1; 
    arma::vec clus_int = conv_to<arma::vec>::from( clus_int_uvec ); 
    arma::uvec clus_inact_uvec = find( nc == 0 ); 
    
    helper::update_xi( current_xi, current_c, z, at_risk, mu_prior, s2_prior, s2_MH, clus_int, J );
    current_xi.rows( clus_inact_uvec ).fill( 0.0 );
    
  }

  void split_merge( arma::vec& current_c,
                    arma::mat& current_xi,
                    arma::vec& nc, // The cluster size
                    const arma::mat z,
                    const arma::mat at_risk,
                    const arma::vec log_Vn,
                    const double theta,
                    const int n, // Number of observations
                    const int Km,
                    const double mu_prior,
                    const double s2_prior,
                    const double s2_MH,
                    const unsigned int nL_split, // Number of restricted Gibbs for launch split
                    const unsigned int nL_merge, // Number of launch state for merge
                    const int J,
                    arma::vec& sm_record,
                    arma::vec& sm_accept,
                    arma::vec& logA_vec,
                    const int t ){
    
    // Placeholder
    arma::vec mu_prior_vec( J, arma::fill::value( mu_prior ) ); 
    arma::mat s2_prior_mat( J, J, arma::fill::eye );
    s2_prior_mat *= s2_prior;
    
    arma::uvec obs_index;
    arma::uvec S; 
    arma::uvec active_clus = find( nc != 0 ) + 1;
    arma::uvec inactive_clus = find( nc == 0 ) + 1;
    arma::vec clus_sm_init( 2, arma::fill::zeros );
    unsigned int Kp = active_clus.size();
    
    arma::vec nc_splitL = nc;
    arma::vec clus_sm_splitL( 2, arma::fill::zeros );
    arma::vec splitL_c = current_c;
    arma::mat splitL_xi = current_xi;
    int is = 0;
    
    arma::vec nc_mergeL = nc;
    arma::vec clus_sm_mergeL( 2, arma::fill::zeros );
    arma::vec mergeL_c = current_c;
    arma::mat mergeL_xi = current_xi;
    arma::vec clus_int_merge( 1, fill::zeros );
    int im = 0;
    
    unsigned int Kp_proposed = 0.0;
    arma::vec nc_proposed = nc;
    arma::vec proposed_c;
    arma::mat proposed_xi;
    double logA = 0.0;
    double logU = log( randu( 1 )[ 0 ] );
    
    // Step 1 - Select two distinct observations
    obs_index = randperm( n, 2 ); // Observation index from 0 to (n-1)
    while( active_clus.size() == Km // The case when Kp = Km
             and ( current_c( obs_index( 0 ) ) == current_c( obs_index( 1 ) ) ) ){
      
      obs_index = randperm( n, 2 );
      
    }
    
    clus_sm_init( 0 ) = current_c( obs_index( 0 ) ); // This is ci
    clus_sm_init( 1 ) = current_c( obs_index( 1 ) ); // This is cj
    
    // Step 2 - Create set S
    S = find( current_c == clus_sm_init( 0 ) or current_c == clus_sm_init( 1 ) ); // In the same group as in i and j
    S = S( find( S != obs_index( 0 ) and S != obs_index( 1 ) ) ); // not include i and j
    
    // Step 3(s) - Split Launch States
    clus_sm_splitL = clus_sm_init;
    if( clus_sm_splitL( 0 ) == clus_sm_splitL( 1 ) ){ // If ci = cj, we propose a new cluster.
      nc_splitL( clus_sm_splitL( 0 ) - 1 ) -= 1;
      clus_sm_splitL( 0 ) = inactive_clus( 0 );
      splitL_c( obs_index( 0 ) ) = inactive_clus( 0 );
      nc_splitL( clus_sm_splitL( 0 ) - 1 ) += 1;
    }
    
    for( int s = 0; s < S.size(); ++s ){
      is = S( s );
      nc_splitL( splitL_c( is ) - 1 ) -= 1;
      splitL_c( is ) = clus_sm_splitL( randu( ) < 0.5 );
      nc_splitL( splitL_c( is ) - 1 ) += 1;
    }
    
    splitL_xi.row( clus_sm_splitL( 0 ) - 1 ) = helper::mvrnormArma( 1, mu_prior_vec, s2_prior_mat ).row( 0 );
    splitL_xi.row( clus_sm_splitL( 1 ) - 1 ) = helper::mvrnormArma( 1, mu_prior_vec, s2_prior_mat ).row( 0 );
    
    for( int ns = 0; ns < nL_split; ++ns ){
      helper::rGibbs( splitL_c, splitL_xi, nc_splitL, z, at_risk, mu_prior, s2_prior, s2_MH, S, clus_sm_splitL, J );
    }
    
    // Step 3(m) - Merge Launch States
    clus_sm_mergeL = clus_sm_init;
    mergeL_xi.row( clus_sm_mergeL( 0 ) - 1 ).fill( 0 );
    nc_mergeL( clus_sm_mergeL( 0 ) - 1 ) -= 1;
    clus_sm_mergeL( 0 ) = clus_sm_mergeL( 1 );
    mergeL_c( obs_index( 0 ) ) = clus_sm_mergeL( 0 );
    nc_mergeL( clus_sm_mergeL( 0 ) - 1 ) += 1;
    
    for( int s = 0; s < S.size(); ++s ){
      im = S( s );
      nc_mergeL( mergeL_c( im ) - 1 ) -= 1;
      mergeL_c( im ) = clus_sm_mergeL( 1 );
      nc_mergeL( mergeL_c( im ) - 1 ) += 1;
    }
    
    mergeL_xi.row( clus_sm_mergeL( 0 ) - 1 ) = helper::mvrnormArma( 1, mu_prior_vec, s2_prior_mat ).row( 0 );
    clus_int_merge( 0 ) = clus_sm_mergeL( 0 );
    
    for( int nm = 0; nm < nL_merge; ++nm ){
      helper::update_xi( mergeL_xi, mergeL_c, z, at_risk, mu_prior, s2_prior, s2_MH, clus_int_merge, J );
    }
    
    // Step 4 (Split) or 5 (Merge) - Propose new c and xi 
    if( clus_sm_init( 0 ) == clus_sm_init( 1 ) ){ // If ci = cj, split
      
      sm_record( t ) = 1;
      
      Kp_proposed = ( Kp + 1 );
      proposed_c = splitL_c;
      proposed_xi = splitL_xi;
      nc_proposed = nc_splitL;
      helper::rGibbs( proposed_c, proposed_xi, nc_proposed, z, at_risk, mu_prior, s2_prior, s2_MH, S, clus_sm_splitL, J );
      
      // Calculate log of acceptance probability
      logA += log_Vn( Kp_proposed - 1 );
      logA += lgamma( nc_proposed( clus_sm_splitL( 0 ) - 1 ) + theta );
      logA += lgamma( nc_proposed( clus_sm_splitL( 1 ) - 1 ) + theta );
      logA -= ( 2 * log( theta ) );
      logA += helper::ldnormARMA( proposed_xi.row( clus_sm_splitL( 0 ) - 1 ), mu_prior, s2_prior, J );
      logA += helper::ldnormARMA( proposed_xi.row( clus_sm_splitL( 1 ) - 1 ), mu_prior, s2_prior, J );

      logA -= log_Vn( Kp - 1 );
      logA -= lgamma( S.size() + 2 + theta );
      logA += log( theta );
      logA -= helper::ldnormARMA( current_xi.row( clus_sm_init( 0 ) - 1 ), mu_prior, s2_prior, J );
      
      for( int i = 0; i < n; ++i ){
        logA += helper::logmar_data( z.row( i ), at_risk.row( i ), proposed_xi.row( proposed_c( i ) - 1 ) );
        logA -= helper::logmar_data( z.row( i ), at_risk.row( i ), current_xi.row( current_c( i ) - 1 ) );
      }
      
      logA -= helper::log_probGS_c( proposed_c, splitL_c, nc_splitL, proposed_xi, z, at_risk, S, clus_sm_splitL );
      
      logA += helper::log_probGS_xi_k( current_xi.row( clus_sm_init( 0 ) - 1 ), mergeL_c, z, at_risk, mu_prior, s2_prior, clus_sm_init( 0 ), J );
      logA -= helper::log_probGS_xi_k( proposed_xi.row( clus_sm_splitL( 0 ) - 1 ), splitL_c, z, at_risk, mu_prior, s2_prior, clus_sm_splitL( 0 ), J );
      logA -= helper::log_probGS_xi_k( proposed_xi.row( clus_sm_splitL( 1 ) - 1 ), splitL_c, z, at_risk, mu_prior, s2_prior, clus_sm_splitL( 1 ), J );
      
    } else { // If ci != cj, merge
      
      sm_record( t ) = 0;
      
      Kp_proposed = ( Kp - 1 );
      proposed_c = mergeL_c;
      proposed_xi = mergeL_xi;
      nc_proposed = nc_mergeL;
      helper::update_xi( proposed_xi, proposed_c, z, at_risk, mu_prior, s2_prior, s2_MH, clus_int_merge, J );
      
      // Calculate log of acceptance probability
      logA += log_Vn( Kp_proposed - 1 );
      logA += lgamma( S.size() + 2 + theta );
      logA -= log( theta );
      logA += helper::ldnormARMA( proposed_xi.row( clus_int_merge( 0 ) - 1 ), mu_prior, s2_prior, J );
      
      logA -= log_Vn( Kp - 1 );
      logA -= lgamma( nc( clus_sm_init( 0 ) - 1 ) + theta );
      logA -= lgamma( nc( clus_sm_init( 1 ) - 1 ) + theta );
      logA += ( 2 * log( theta ) );
      logA -= helper::ldnormARMA( current_xi.row( clus_sm_init( 0 ) - 1 ), mu_prior, s2_prior, J );
      logA -= helper::ldnormARMA( current_xi.row( clus_sm_init( 1 ) - 1 ), mu_prior, s2_prior, J );

      for( int i = 0; i < n; ++i ){
        logA += helper::logmar_data( z.row( i ), at_risk.row( i ), proposed_xi.row( proposed_c( i ) - 1 ) );
        logA -= helper::logmar_data( z.row( i ), at_risk.row( i ), current_xi.row( current_c( i ) - 1 ) );
      }
      
      logA += helper::log_probGS_c( current_c, splitL_c, nc_splitL, current_xi, z, at_risk, S, clus_sm_init );
      logA += helper::log_probGS_xi_k( current_xi.row( clus_sm_init( 0 ) - 1 ), splitL_c, z, at_risk, mu_prior, s2_prior, clus_sm_init( 0 ), J );
      logA += helper::log_probGS_xi_k( current_xi.row( clus_sm_init( 1 ) - 1 ), splitL_c, z, at_risk, mu_prior, s2_prior, clus_sm_init( 1 ), J );
      logA -= helper::log_probGS_xi_k( proposed_xi.row( clus_sm_mergeL( 0 ) - 1 ), mergeL_c, z, at_risk, mu_prior, s2_prior, clus_sm_mergeL( 0 ), J );
      
    }
    
    // Determine
    if( logU < logA ){
      current_c = proposed_c;
      current_xi = proposed_xi;
      nc = nc_proposed;
      sm_accept( t ) = 1;
    } else {
      sm_accept( t ) = 0;
    }
    
    logA_vec( t ) = logA;
    
    arma::uvec clus_inact_uvec = find( nc == 0 ); 
    current_xi.rows( clus_inact_uvec ).fill( 0.0 );
    
  }

}

// [[Rcpp::export]]
arma::vec log_Vn_TB( double Km, double pi, double N, double theta ){
  
  // Placeholder
  arma::vec lVn( Km, fill::zeros );
  double max_lVn_k = 0.0;
  
  for( int kp = 1; kp <= Km; ++kp ){
    
    arma::vec log_Vn_kp( Km - kp + 1, fill::zeros );
    unsigned int loc = 0;
    
    for( int k = kp; k <= Km; ++k ){
      
      log_Vn_kp( loc ) += lgamma( k + 1 );
      log_Vn_kp( loc ) -= lgamma( ( k - kp ) + 1 );
      log_Vn_kp( loc ) += lgamma( theta * k );
      log_Vn_kp( loc ) -= lgamma( ( theta * k ) + N );
      log_Vn_kp( loc ) += helper::log_pTB( k, Km, pi );
      
      loc += 1;
      
    }
    
    // Apply log-sum-exp trick
    max_lVn_k = max( log_Vn_kp );
    lVn( kp - 1 ) = max_lVn_k + log( accu( exp( log_Vn_kp  - max_lVn_k ) ) );
    
  }
  
  return lVn;
  
}

// [[Rcpp::export]]
void mod_sm( arma::vec& c_init, arma::vec& nc_init, arma::mat& xi_init, arma::mat& ar_init, // Initialization
             arma::mat& c_store, arma::cube& xi_store, arma::mat& nc_store, // Placeholder for store the result
             const arma::mat z, const arma::mat zero_loc, const arma::vec logVn, // Data
             const unsigned int Km, const double a_ar, const double b_ar, 
             const double mu_prior, const double s2_prior, const double theta,
             unsigned int iter, unsigned int adaptive_thres, const double s2_MH,
             const unsigned int nL_split, const unsigned int nL_merge,
             arma::vec& split_or_merge, arma::vec& sm_accept, arma::vec& logA_sm ){
  
  unsigned int n = z.n_rows;
  unsigned int J = z.n_cols;
  unsigned int n0 = accu( z == 0 );
  arma::vec obs_index;
  obs_index = regspace( 0, n - 1 );
  
  arma::vec n_existed( Km, fill::ones );
  arma::cube cMat( J, J, Km, fill::zeros );
  arma::mat xbarvec( Km, J, fill::zeros );
  
  for( int it = 0; it < iter; ++it ){
    
    // Run the sampler
    sampler::update_at_risk( ar_init, z, xi_init, c_init, a_ar, b_ar, zero_loc, n0 );
    sampler::update_xi_ADAP( xi_init, z, c_init, ar_init, mu_prior, s2_prior, J, cMat, n_existed, adaptive_thres );
    helper::cov_update( cMat, xbarvec, n_existed, xi_init, c_init );
    helper::mean_update( xbarvec, n_existed, xi_init, c_init, Km );
    sampler::split_merge( c_init, xi_init, nc_init, z, ar_init, logVn, theta, n, Km, mu_prior, s2_prior, s2_MH, nL_split, nL_merge, J, split_or_merge, sm_accept, logA_sm, it );
    // sampler::alg8( c_init, xi_init, nc_init, Km, z, ar_init, logVn, n, J, theta, mu_prior, s2_prior, s2_MH );
    
    // Record the result
    c_store.col( it ) = c_init;
    xi_store.slice( it ) = xi_init;
    nc_store.col( it ) = nc_init;
    
  }
  
}

// [[Rcpp::export]]
void mod( arma::vec& c_init, arma::vec& nc_init, arma::mat& xi_init, arma::mat& ar_init, // Initialization
          arma::mat& c_store, arma::cube& xi_store, arma::mat& nc_store, // Placeholder for store the result
          const arma::mat z, const arma::mat zero_loc, const arma::vec logVn, // Data
          const unsigned int Km, const double a_ar, const double b_ar, 
          const double mu_prior, const double s2_prior, const double theta,
          unsigned int iter, unsigned int adaptive_thres, const double s2_MH,
          const unsigned int nL_split, const unsigned int nL_merge,
          arma::vec& split_or_merge, arma::vec& sm_accept, arma::vec& logA_sm ){
  
  unsigned int n = z.n_rows;
  unsigned int J = z.n_cols;
  unsigned int n0 = accu( z == 0 );
  arma::vec obs_index;
  obs_index = regspace( 0, n - 1 );
  
  arma::vec n_existed( Km, fill::ones );
  arma::cube cMat( J, J, Km, fill::zeros );
  arma::mat xbarvec( Km, J, fill::zeros );
  
  for( int it = 0; it < iter; ++it ){
    
    // Run the sampler
    sampler::update_at_risk( ar_init, z, xi_init, c_init, a_ar, b_ar, zero_loc, n0 );
    sampler::update_xi_ADAP( xi_init, z, c_init, ar_init, mu_prior, s2_prior, J, cMat, n_existed, adaptive_thres );
    helper::cov_update( cMat, xbarvec, n_existed, xi_init, c_init );
    helper::mean_update( xbarvec, n_existed, xi_init, c_init, Km );
    sampler::split_merge( c_init, xi_init, nc_init, z, ar_init, logVn, theta, n, Km, mu_prior, s2_prior, s2_MH, nL_split, nL_merge, J, split_or_merge, sm_accept, logA_sm, it );
    sampler::alg8( c_init, xi_init, nc_init, Km, z, ar_init, logVn, n, J, theta, mu_prior, s2_prior, s2_MH );
    
    // Record the result
    c_store.col( it ) = c_init;
    xi_store.slice( it ) = xi_init;
    nc_store.col( it ) = nc_init;
    
  }
  
}

