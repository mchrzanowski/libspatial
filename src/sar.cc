#include "sar.h"
#include <cmath>

using namespace arma;

SAR::SAR(const colvec &y, const mat &X, const sp_mat &W) :
  ARModel(y, X, W), XT(X.t()), I(speye<sp_mat>(W.n_rows, W.n_cols)) {}

void 
SAR::solve(){
  calculate_rho();
  const sp_mat A = I - rho * W;
  const mat XTA = XT * A;
  const colvec Ay = A * y;
  if (! arma::solve(beta, XTA * XTA.t(), XTA * Ay)){
    beta = zeros<colvec>(X.n_cols);
  }
  const colvec val = Ay - (beta.t() * XTA).t();
  sigma_sq = dot(val, val) / m;
}

double
SAR::log_likelihood() {
  const sp_mat A = I - rho * W;
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
  const sp_mat A = I - rho_hat * W;
  const colvec Ay = A * y;
  const mat XTA = XT * A;

  colvec beta_hat;
  if (! arma::solve(beta_hat, XTA * XTA.t(), XTA * Ay)){
    beta_hat = zeros<colvec>(X.n_cols);
  }
  const colvec val = Ay - (beta_hat.t() * XTA).t();
  return (-2. / m) * calc_log_det(rho_hat) + log(dot(val, val));
}
