#include "car.h"
#include <cmath>

using namespace arma;

CAR::CAR(const colvec &y, const mat &X, const sp_mat &W) :
  ARModel(y, X, W), XT_W(X.t() * W) {}

void
CAR::solve(){
  calculate_rho();
  const sp_mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho * W;
  const mat XT_A = X.t() - rho * XT_W;
  if (! arma::solve(beta, XT_A * X, XT_A * y) || ! is_finite(beta)){
    beta = zeros<colvec>(X.n_cols);
  }
  const colvec y_XB = y - X * beta;
  sigma_sq = dot(y_XB, A * y_XB) / m;
}

/* a version of the log-likelihood parameterized by \rho only
and not by \beta or \sigma^2 .*/
double
CAR::rho_ll(double rho_hat){
  const sp_mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho_hat * W;
  const mat XT_A = X.t() - rho_hat * XT_W;
  colvec beta_hat;
  if (! arma::solve(beta_hat, XT_A * X, XT_A * y) || ! is_finite(beta)){
    beta_hat = zeros<colvec>(X.n_cols);
  }
  const colvec diff = y - X * beta_hat;
  return (-2 / m) * calc_log_det(rho_hat) / 2 + log(dot(diff, A * diff));
}

double
CAR::log_likelihood(){
  const sp_mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho * W;
  const colvec y_XB = y - X * beta;
  return calc_log_det(rho) / 2 - dot(y_XB, A * y_XB) / (2 * sigma_sq)
    - m * (log(2 * datum::pi) + log(sigma_sq)) / 2;
}
