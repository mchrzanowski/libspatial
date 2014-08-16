#include "sar.h"
#include <cmath>

using namespace arma;

SAR::SAR(const colvec &y, const mat &X, const sp_mat &W) : ARModel(y, X, W),
  XT(X.t()), XT_X(XT * X),
  IX(speye<sp_mat>(X.n_rows, X.n_rows) - X * pinv(XT_X) * XT),
  W_y(W * y), XT_W_y(XT * W_y), XT_y(XT * y) {}

void 
SAR::solve(){
  calculate_rho();
  if (! arma::solve(beta, XT_X, XT_y - rho * XT_W_y) || ! is_finite(beta)){
    beta = zeros<colvec>(X.n_cols);
  }
  const colvec IX_A_y = IX * (y - rho * W_y);
  sigma_sq = dot(IX_A_y, IX_A_y) / m;
}

double
SAR::log_likelihood() {
  const colvec val = y - rho * W_y - X * beta;
  // log_det(A) = \sum_{i=1}^n \log(1 - \rho\lambda_i)
  static const colvec eigs = eig_sym(mat(W));
  return sum(log(1 - rho * eigs)) - dot(val, val) / (2 * sigma_sq)
    - m * log(2 * datum::pi) / 2 - m * log(sigma_sq) / 2;
}

/* a version of the log-likelihood parameterized by \rho only
and not by \beta or \sigma^2 .*/
double
SAR::rho_ll(double rho_hat){
  const colvec val = IX * (y - rho_hat * W_y);
  return (-2. / m) * calc_log_det(rho_hat) + log(dot(val, val));
}
