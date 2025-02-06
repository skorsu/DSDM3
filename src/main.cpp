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
    double y = lse( lprob );
    arma::vec prob = exp( lprob - y );
    prob.clean( pow( 10, -20 ) );
    return normalise( prob, 1 );
  }
  
  double logVn_TB( unsigned int Kp, unsigned int Km,
                   unsigned int N, double theta, double pi ){
    arma::vec k = regspace( Kp, Km );
    arma::vec sum_k = ( k * log( pi ) ) + ( ( Km - k ) * log( 1 - pi ) ) + ( lgamma( k * theta ) );
    sum_k -= ( lgamma( ( k - Kp ) + 1 ) + lgamma( ( theta * k ) + N ) + lgamma( ( Km - k ) + 1 ) );
    return lse( sum_k ) + lgamma( Km + 1 ) - log( 1 - pow( 1 - pi, Km ) );
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
  
  arma::vec logVn_TB_seq( unsigned int Km, unsigned int N, double theta, double pi ){
    
    arma::vec result( Km, fill::zeros );
    for( int k = 0; k < Km; ++k ){
      result[ k ] = helper::logVn_TB( k + 1, Km, N, theta, pi );
    }
    
    return result;
  }
  
  void update_xi( arma::mat& current_xi,
                  const arma::mat z,
                  const arma::vec c,
                  const arma::mat at_risk,
                  const arma::vec active_index, 
                  const double mu_prior,
                  const double s2_prior,
                  const double J ){
    
    unsigned int Kp = active_index.size();
    
    // Create the covariance matrix for MH. 
    arma::mat s2_mat( J, J, fill::eye );
    s2_mat *= pow( 10, -3 );
    
    // Update xi for each active cluster
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
      proposed_xi_k = helper::mvrnormArma( 1, current_xi_k.t(), s2_mat ).row( 0 ); // Propose new xi
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
  
  void restricted_Gibbs( arma::vec& c, 
                         arma::mat& xi, 
                         const arma::vec clus_int, // Cluster that go into restricted Gibbs 
                         unsigned int i, unsigned int j,
                         const arma::mat z,
                         const arma::mat at_risk,
                         const double theta,
                         const double mu_prior, // For MH update
                         const double s2_prior,
                         const double J ) { // For MH update
    
    // Obtain the cluster size
    unsigned int n = c.size(); 
    arma::vec ni( 2, arma::fill::zeros );
    for( int ii = 0; ii < n; ++ii ){
      if( ii != i and ii != j and ( c( ii ) == clus_int( 0 ) or c( ii ) == clus_int( 1 ) ) ){
        ni( find( c( ii ) == clus_int ) ) += 1;
      }
    }
    
    // Reallocate
    arma::vec lprob( 2, arma::fill::zeros );
    double prob0 = 0.0;
    double u = 0.0;
    for( int ii = 0; ii < n; ++ii ){
      if( ii != i and ii != j and ( c( ii ) == clus_int( 0 ) or c( ii ) == clus_int( 1 ) ) ){
        ni( find( c( ii ) == clus_int ) ) -= 1;
        lprob = log( ni + theta );
        lprob( 0 ) += helper::logmar_data( z.row( ii ), at_risk.row( ii ), xi.row( clus_int( 0 ) - 1 ) );
        lprob( 1 ) += helper::logmar_data( z.row( ii ), at_risk.row( ii ), xi.row( clus_int( 1 ) - 1 ) );
        prob0 = helper::norm_prob( lprob )( 0 );
        u = randu( 1 )( 0 ); 
        c( ii ) = clus_int( u < prob0 );
        ni( find( c( ii ) == clus_int ) ) += 1;
      }
    }
    
    // Update xi
    helper::update_xi( xi, z, c, at_risk, clus_int, mu_prior, s2_prior, J );
    
  }
  
  double probGS( unsigned int index_ci_a, // 0 or 1 only
                 unsigned int index_ci_b, // 0 or 1 only 
                 arma::vec n_size_b, // vector with 2 elements
                 const arma::rowvec zi,
                 const arma::rowvec ar_i,
                 const arma::rowvec xi_0,
                 const arma::rowvec xi_1 ){
    
    // Note: This calculates p( ci_a | ci_b )
    
    arma::vec adjust_vec( 2, fill::zeros );
    adjust_vec( index_ci_b ) -= 1;
    arma::vec lprob_GS( 2, fill::zeros );
    
    lprob_GS( 0 ) = log( n_size_b( 0 ) + adjust_vec( 0 ) ) + helper::logmar_data( zi, ar_i, xi_0 );
    lprob_GS( 1 ) = log( n_size_b( 1 ) + adjust_vec( 1 ) ) + helper::logmar_data( zi, ar_i, xi_1 );
    
    arma::vec prob_GS = helper::norm_prob( lprob_GS );
    
    return prob_GS( index_ci_a );
    
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
  
  void split_merge( arma::vec& current_c,
                    arma::mat& current_xi,
                    const arma::mat z,
                    const arma::mat at_risk,
                    const unsigned int n,
                    const double J,
                    const arma::vec index_obs, // Index of the observation from 0 to n-1
                    const unsigned int Km,
                    const double theta,
                    const unsigned int nL_split, // Number of launch state for split
                    const unsigned int nL_merge, // Number of launch state for split
                    const arma::vec log_Vn, 
                    const double mu_prior,
                    const double s2_prior ){
    
    arma::vec mu_vec( J, fill::value( mu_prior ) );
    arma::mat s2_mat( J, J, fill::eye );
    s2_mat *= s2_prior;
    
    // Obtain the cluster size
    arma::vec ni( Km, arma::fill::zeros );
    for( int i = 0; i < n; ++i ){
      ni( current_c( i ) - 1 ) += 1;
    } 
    
    unsigned int Kp = accu( ni != 0 );
    arma::uvec inactive_index_hold = find( ni == 0 );
    
    // Select two observations
    arma::uvec index_choose = randperm( n, 2 );
    arma::vec c_choose = current_c( index_choose );
    
    // Put a constrain when Kp = Km. We can only perform merge step.
    if( Kp == Km ){
      while( c_choose( 0 ) == c_choose( 1 ) ){
        index_choose = randperm( n, 2 );
        c_choose = current_c( index_choose );
      }
    }
    
    // Define set S
    arma::uvec S = find( ( current_c == c_choose( 0 ) or current_c == c_choose( 1 ) ) and index_obs != index_choose( 0 ) and index_obs != index_choose( 1 ) );
    unsigned int nS = S.size();
    
    // Define Launch State
    arma::vec launch_split_c = current_c;
    arma::mat launch_split_xi = current_xi;
    arma::vec split_c_choose = c_choose; 
    arma::vec launch_merge_c = current_c;
    arma::mat launch_merge_xi = current_xi; 
    arma::vec merge_c_choose( 2, arma::fill::value( c_choose( 1 ) ) );
    
    // Split Launch State
    if( Kp < Km ){
      unsigned int new_clus = inactive_index_hold( 0 ) + 1;
      split_c_choose( 0 ) = new_clus;
      launch_split_c( index_choose( 0 ) ) = new_clus;
      launch_split_xi.row( new_clus - 1 ) = helper::mvrnormArma( 1, mu_vec, s2_mat ).row( 0 );
      launch_split_c.rows( S ) = split_c_choose( randu( nS ) < 0.5 );
    }
    
    for( int l = 0; l < nL_split; ++l ){
      helper::restricted_Gibbs( launch_split_c, launch_split_xi, split_c_choose, index_choose( 0 ), index_choose( 1 ), z, at_risk, theta, mu_prior, s2_prior, J );
    }
    
    arma::vec ni_lsplit( 2, arma::fill::zeros );
    for( int kk = 0; kk < 2; ++kk ){
      ni_lsplit( kk ) += accu( launch_split_c == split_c_choose( kk ) );
    }
    
    // Merge Launch State
    launch_merge_c( index_choose( 0 ) ) = merge_c_choose( 0 );
    launch_merge_xi.row( merge_c_choose( 1 ) - 1 ) = helper::mvrnormArma( 1, mu_vec, s2_mat ).row( 0 );
    launch_merge_c.rows( S ).fill( merge_c_choose( 1 ) );
    for( int l = 0; l < nL_merge; ++l ){
      helper::update_xi( launch_merge_xi, z, launch_merge_c, at_risk, merge_c_choose, mu_prior, s2_prior, J );
    }
    
    // Perform Split-Merge ( Propose new c and xi )
    arma::vec proposed_c;
    arma::mat proposed_xi;
    double logA = 0.0;
    double logU = log( randu( 1 )( 0 ) );
    arma::vec ni_split( Km, arma::fill::zeros );
    arma::uvec Sij = join_vert( S, index_choose );
    unsigned int nSij = Sij.size();
    unsigned int s;
    arma::uvec ia;
    arma::uvec ib;
    
    if( c_choose( 0 ) == c_choose( 1 ) and Kp < Km ){ // Split 
      
      proposed_c = launch_split_c;
      proposed_xi = launch_split_xi;
      helper::restricted_Gibbs( proposed_c, proposed_xi, split_c_choose, index_choose( 0 ), index_choose( 1 ), z, at_risk, theta, mu_prior, s2_prior, J );
      
      logA += log_Vn( Kp - 1 );
      logA += lgamma( accu( proposed_c == split_c_choose( 0 ) ) );
      logA += lgamma( accu( proposed_c == split_c_choose( 1 ) ) );
      // logA += ldnormARMA( proposed_xi.row( split_c_choose( 0 ) - 1 ), mu_prior, s2_prior, J );
      // logA += ldnormARMA( proposed_xi.row( split_c_choose( 1 ) - 1 ), mu_prior, s2_prior, J );
      logA -= log_Vn( Kp - 2 );
      logA -= lgamma( accu( current_c == c_choose( 0 ) ) );
      // logA -= ldnormARMA( current_xi.row( c_choose( 1 ) - 1 ), mu_prior, s2_prior, J );
      
      for( int ii = 0; ii < nSij; ++ii ){
        s = Sij( ii );
        logA += helper::logmar_data( z.row( s ), at_risk.row( s ), proposed_xi.row( proposed_c( s ) - 1 ) );
        logA -= ( current_c( s ) == c_choose( 0 ) ) * helper::logmar_data( z.row( s ), at_risk.row( s ), current_xi.row( current_c( s ) - 1 ) );
        ia = find( proposed_c( s ) == split_c_choose );
        ib = find( launch_split_c( s ) == split_c_choose );
        logA -= helper::probGS( ia( 0 ), ib( 0 ), ni_lsplit, z.row( s ), at_risk.row( s ), launch_split_xi.row( split_c_choose( 0 ) - 1 ), launch_split_xi.row( split_c_choose( 1 ) - 1 ) );
      }
      
    } else { // Merge
      
      proposed_c = launch_merge_c;
      proposed_xi = launch_merge_xi;
      helper::update_xi( proposed_xi, z, proposed_c, at_risk, merge_c_choose, mu_prior, s2_prior, J );
      
      logA += log_Vn( Kp - 2 );
      logA += lgamma( accu( proposed_c == merge_c_choose( 0 ) ) );
      logA -= log_Vn( Kp - 1 );
      logA -= lgamma( accu( current_c == c_choose( 0 ) ) );
      logA += lgamma( accu( current_c == c_choose( 1 ) ) );
      
      for( int ii = 0; ii < nSij; ++ii ){
        s = Sij( ii );
        logA += helper::logmar_data( z.row( s ), at_risk.row( s ), proposed_xi.row( proposed_c( s ) - 1 ) );
        logA -= helper::logmar_data( z.row( s ), at_risk.row( s ), current_xi.row( current_c( s ) - 1 ) );
        ia = find( current_c( s ) == c_choose );
        ib = find( launch_split_c( s ) == split_c_choose );
        logA += helper::probGS( ia( 0 ), ib( 0 ), ni_lsplit, z.row( s ), at_risk.row( s ), launch_split_xi.row( split_c_choose( 0 ) - 1 ), launch_split_xi.row( split_c_choose( 1 ) - 1 ) );
      }
      
    }
    
    // Determine
    if( logU < logA ){
      current_c = proposed_c;
      current_xi = proposed_xi;
    }
    
  }
  
  void alg8( arma::vec& current_c,
             arma::mat& current_xi,
             unsigned int Km,
             const arma::mat z,
             const arma::mat at_risk,
             const arma::vec log_Vn,
             const unsigned int n,
             const unsigned int J,
             const double theta,
             const double mu_prior,
             const double s2_prior ){
    
    unsigned int Kp = 0;
    int m = 5;
    arma::uvec active_index;
    arma::uvec inactive_index_hold;
    unsigned int proposed_index = 0;
    arma::vec lprob_realloc( Km, fill::zeros );
    arma::vec lprob_active; 
    arma::vec prob_active;
    IntegerVector hold;
    int c_index_f;
    arma::vec mu_vec( J, fill::value( mu_prior ) );
    arma::mat s2_mat( J, J, fill::eye );
    s2_mat *= s2_prior;
    
    // Obtain the cluster size
    arma::vec ni( Km, arma::fill::zeros );
    for( int i = 0; i < n; ++i ){
      ni( current_c( i ) - 1 ) += 1;
    }
    
    // Reallocation
    for( int i = 0; i < n; ++i ){
      lprob_realloc.zeros();
      proposed_index = 0;
      ni( current_c( i ) - 1 ) -= 1;
      Kp = accu( ni != 0 );
      m = ( Kp < Km );
      if( ni( current_c( i ) - 1 ) > 0 and m == 1 ){
        Kp += 1;
        inactive_index_hold = find( ni == 0 );
        proposed_index = inactive_index_hold( 0 ) + 1;
        current_xi.row( proposed_index - 1 ) = helper::mvrnormArma( 1, mu_vec, s2_mat ).row( 0 );
      } else if( ni( current_c( i ) - 1 ) == 0 ){
        proposed_index = current_c( i );
      }
      if( proposed_index != 0 ){
        lprob_realloc( proposed_index - 1 ) += log_Vn( Kp - 1 );
        lprob_realloc( proposed_index - 1 ) -= log_Vn( Kp - 2 );
        lprob_realloc( proposed_index - 1 ) += log( theta );
        lprob_realloc( proposed_index - 1 ) += helper::logmar_data( z.row( i ), at_risk.row( i ), current_xi.row( proposed_index - 1 ) );
      }
      active_index = find( ni != 0 ) + 1;
      for( int kk = 0; kk < active_index.size(); ++kk ){
        lprob_realloc( active_index( kk ) - 1 ) += helper::logmar_data( z.row( i ), at_risk.row( i ), current_xi.row( active_index( kk ) - 1 ) );
        lprob_realloc( active_index( kk ) - 1 ) += log( ni( active_index( kk ) - 1 ) + theta );
      }
      active_index = find( lprob_realloc != 0.0 ) + 1;
      prob_active = helper::norm_prob( lprob_realloc.rows( active_index - 1 ) );
      hold = helper::myseq( 0, prob_active.size() - 1 );
      c_index_f = helper::sample_prob_cpp( hold, wrap( prob_active ) );
      current_c( i ) = active_index( c_index_f );
      ni( current_c( i ) - 1 ) += 1;
      
    }
    
    // Update xi
    IntegerVector clus_ii = wrap( find( ni != 0 ) );
    arma::vec clus_int = helper::convertIntegerVectorToArmaVec( clus_ii );
    helper::update_xi( current_xi, z, current_c, at_risk, clus_int, mu_prior, s2_prior, J );
    
    // Optional - Clean up xi matrix
    arma::uvec clus_ni = find( ni == 0 );
    current_xi.rows( clus_ni ).fill( 0.0 );
    
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

}

// [[Rcpp::export]]
void mod( arma::mat c_result,
          arma::cube xi_result,
          const unsigned int Km,
          const double theta,
          const double pi_lambda,
          const double a_gamma,
          const double b_gamma,
          const double mu_prior,
          const double s2_prior,
          const int adaptive_thres,
          const unsigned int nL_split,
          const unsigned int nL_merge,
          const arma::mat z,
          const arma::mat z_zero_index,
          const unsigned int iter ){
  
  // Pre-define settings
  unsigned int n = z.n_rows;
  unsigned int J = z.n_cols;
  unsigned int n0 = accu( z == 0 );
  arma::vec obs_index;
  obs_index = regspace( 0, n - 1 );
  
  // Calculated Result
  arma::vec n_existed( Km, fill::ones );
  arma::cube cMat( J, J, Km, fill::zeros );
  arma::mat xbarvec( Km, J, fill::zeros );
  arma::vec log_Vn = helper::logVn_TB_seq( Km, n, theta, pi_lambda );
  
  // Intermediate Result
  arma::mat ar_mat( n, J, fill::ones );
  arma::mat xi_mat( Km, J, fill::zeros );
  arma::vec c_vec( n, fill::ones );
  
  for( int it = 0; it < iter; ++it ){
    
    // Update parameters
    sampler::update_at_risk( ar_mat, z, xi_mat, c_vec, a_gamma, b_gamma, z_zero_index, n0 );
    sampler::update_xi_ADAP( xi_mat, z, c_vec, ar_mat, mu_prior, s2_prior, J, cMat, n_existed, adaptive_thres );
    helper::cov_update( cMat, xbarvec, n_existed, xi_mat, c_vec );
    helper::mean_update( xbarvec, n_existed, xi_mat, c_vec, Km );
    sampler::split_merge( c_vec, xi_mat, z, ar_mat, n, J, obs_index, Km, theta, nL_split, nL_merge, log_Vn, mu_prior, s2_prior );
    sampler::alg8( c_vec, xi_mat, Km, z, ar_mat, log_Vn, n, J, theta, mu_prior, s2_prior );
    
    // Store the result
    xi_result.slice( it ) = xi_mat;
    c_result.col( it ) = c_vec;
    
  }
  
}
