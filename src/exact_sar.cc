#include "exact_sar.h"
#include <iostream>

using namespace arma;

ExactSAR::ExactSAR(const mat &W, const mat &X, const colvec &y) :
  SAR(y, X.n_rows, X.n_cols), W(W), X(X),
  IX(speye<sp_mat>(X.n_rows, X.n_rows) - X * pinv(X.t() * X) * X.t()),
  eigs(eig_sym(W)) {}

void
ExactSAR::solve(){
  calculate_rho();
  const mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho * W;
  const colvec A_y = A * y;
  if (! arma::solve(beta, X.t() * X, X.t() * A_y) || ! is_finite(beta)){
    beta = zeros<colvec>(X.n_cols);  
  }
  const colvec IX_A_y = IX * A_y;
  sigma_sq = dot(IX_A_y, IX_A_y) / m;
}

double
ExactSAR::rho_ll(double rho_hat){
  const mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho_hat * W;
  const colvec IX_A_y = IX * A * y;
  // log_det(A) = \sum_{i=1}^n \log(1 - \rho\lambda_i)
  return (-2. / m) * sum(log(1 - rho_hat * eigs)) + log(dot(IX_A_y, IX_A_y));
}

double
ExactSAR::log_likelihood(){
  const mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho * W;
  const colvec A_y = A * y;
  colvec val = A_y - X * beta;
  return sum(log(1 - rho * eigs)) - dot(val, val) / (2 * sigma_sq)
          - m * log(2 * datum::pi) / 2 - m * log(sigma_sq) / 2;
}
