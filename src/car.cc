#include "car.h"
#include <cmath>

using namespace arma;

CAR::CAR(const colvec &y, const mat &X, const sp_mat &W) : ARModel(y, X, W),
  XT(X.t()), XT_X(XT * X), XT_W(XT * W), XT_W_X(XT_W * X), XT_y(XT * y) {}

void
CAR::solve(){
  calculate_rho();
  if (! arma::solve(beta, XT_X - rho * XT_W_X, XT_y - rho * XT_W * y)){
      beta = zeros<colvec>(X.n_cols);
  }
  const colvec y_XB = y - X * beta;
  sigma_sq = dot(y_XB, y_XB - rho * W * y_XB) / m;
}

/* a version of the log-likelihood parameterized by \rho only
and not by \beta or \sigma^2 .*/
double
CAR::rho_ll(double rho_hat){
  colvec beta_hat;
  if (! arma::solve(beta_hat, XT_X - rho * XT_W_X, XT_y - rho * XT_W * y)){
      beta_hat = zeros<colvec>(X.n_cols);
  }
  const colvec diff = y - X * beta_hat;
  return (-2 / m) * calc_log_det(rho_hat) / 2 + 
    log(dot(diff, diff - rho_hat * W * diff));
}

double
CAR::log_likelihood(){
  const sp_mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho * W;
  const colvec y_XB = y - X * beta;
  static const colvec eigs = eig_sym(mat(W));
  return sum(log(1 - rho * eigs)) / 2 - dot(y_XB, A * y_XB) / (2 * sigma_sq)
    - m * (log(2 * datum::pi) + log(sigma_sq)) / 2;
}
