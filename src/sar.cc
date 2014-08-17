#include "sar.h"
#include <cmath>

using namespace arma;

SAR::SAR(const colvec &y, const mat &X, const sp_mat &W) :
  ARModel(y, X, W), XT(X.t()) {}

void 
SAR::solve(){
  calculate_rho();
  const sp_mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho * W;
  const mat XTAA = XT * A * A;
  if (! arma::solve(beta, XTAA * X, XTAA * y)
    || ! is_finite(beta)){
    beta = zeros<colvec>(X.n_cols);
  }
  const colvec val = A * (y - X * beta);
  sigma_sq = dot(val, val) / m;
}

double
SAR::log_likelihood() {
  const sp_mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho * W;
  const colvec val = A * (y - X * beta);
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
  const mat XTAA = XT * A * A;
  colvec beta_hat;
  if (! arma::solve(beta_hat, XTAA * X, XTAA * y)
    || ! is_finite(beta)){
    beta_hat = zeros<colvec>(X.n_cols);
  }
  const colvec val = A * (y - X * beta_hat);
  return (-2. / m) * calc_log_det(rho_hat) + log(dot(val, val));
}
