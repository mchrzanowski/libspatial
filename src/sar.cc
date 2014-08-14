#include "sar.h"
#include <cmath>

using namespace arma;

SAR::SAR(const colvec &y, const mat &X, const sp_mat &W) :
  ARModel(y, X, W),
  IX(speye<sp_mat>(X.n_rows, X.n_rows) - X * pinv(X.t() * X) * X.t()) {}

void 
SAR::solve(){
  calculate_rho();
  const sp_mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho * W;
  const colvec A_y = A * y;
  if (! arma::solve(beta, X.t() * X, X.t() * A_y) || ! is_finite(beta)){
    beta = zeros<colvec>(X.n_cols);  
  }
  const colvec IX_A_y = IX * A_y;
  sigma_sq = dot(IX_A_y, IX_A_y) / m;
}

double
SAR::log_likelihood() {
  const sp_mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho * W;
  const colvec A_y = A * y;
  const colvec val = A_y - X * beta;
  // log_det(A) = \sum_{i=1}^n \log(1 - \rho\lambda_i)
  static const colvec eigs = eig_sym(mat(W));
  return sum(log(1 - rho * eigs)) - dot(val, val) / (2 * sigma_sq)
          - m * log(2 * datum::pi) / 2 - m * log(sigma_sq) / 2;
}

/* a version of the log-likelihood parameterized by \rho only
and not by \beta or \sigma^2 .*/
double
SAR::rho_ll(double rho_hat){
  const sp_mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho_hat * W;
  const colvec IX_A_y = IX * A * y;
  return (-2. / m) * calc_log_det(rho_hat) + log(dot(IX_A_y, IX_A_y));
}
