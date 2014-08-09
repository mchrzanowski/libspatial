#include "exact_sar.h"

using namespace arma;

ExactSAR::ExactSAR(const colvec &y, const mat &X, const sp_mat &W) :
  SAR(y, X, W), eigs(eig_sym(mat(W))) {}

double
ExactSAR::rho_ll(double rho_hat){
  const sp_mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho_hat * W;
  const colvec IX_A_y = IX * A * y;
  // log_det(A) = \sum_{i=1}^n \log(1 - \rho\lambda_i)
  return (-2. / m) * sum(log(1 - rho_hat * eigs)) + log(dot(IX_A_y, IX_A_y));
}

double
ExactSAR::log_likelihood(){
  const sp_mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho * W;
  const colvec A_y = A * y;
  colvec val = A_y - X * beta;
  return sum(log(1 - rho * eigs)) - dot(val, val) / (2 * sigma_sq)
          - m * log(2 * datum::pi) / 2 - m * log(sigma_sq) / 2;
}
