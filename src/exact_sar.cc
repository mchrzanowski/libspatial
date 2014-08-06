#include "exact_sar.h"

using namespace arma;

ExactSAR::ExactSAR(const mat &W, const mat &X, const colvec &y) :
  SAR(y, X.n_rows, X.n_cols), W(W), X(X),
  IX(speye<sp_mat>(X.n_rows, X.n_cols) - X * pinv(X.t() * X) * X.t()) {
    eigs = eig_sym(W);
}

void
ExactSAR::solve(){
  rho = bisection(-1 + datum::eps, 1 - datum::eps);
  const mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho * W;
  const colvec Ay = A * y;
  beta = arma::solve(X.t() * X, X.t() * Ay);
  const colvec IX_A_y = IX * Ay;
  sigma_sq = dot(IX_A_y, IX_A_y) / m;
}

double
ExactSAR::log_likelihood(double rho_hat){
  const mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho_hat * W;
  const colvec IX_A_y = IX * A * y;
  // log_det(A) = \sum_{i=1}^n \log(1 - \rho\lambda_i)
  const double log_det = sum(log(1 - rho_hat * eigs));
  return (-2. / m) * log_det + log(dot(IX_A_y, IX_A_y));
}
