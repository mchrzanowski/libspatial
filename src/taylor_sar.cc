#include "taylor_sar.h"

using namespace arma;

TaylorSAR::TaylorSAR(const sp_mat &W, const mat &X, const colvec &y) :
  SAR(y, X, W) {}

double
TaylorSAR::log_likelihood(){
  const sp_mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho * W;
  const colvec A_y = A * y;
  colvec val = A_y - X * beta;
  static const colvec eigs = eig_sym(mat(W));
  return sum(log(1 - rho * eigs)) - dot(val, val) / (2 * sigma_sq) 
          - m * log(2 * datum::pi) / 2 - m * log(sigma_sq) / 2;
}

double
TaylorSAR::rho_ll(double rho_hat){
  const sp_mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho_hat * W;
  const colvec IX_A_y = IX * A * y;
  double log_det = 0;
  mat W_pow = eye<mat>(W.n_rows, W.n_cols);
  double rho_hat_pow = 1;
  for (unsigned k = 1; k <= 30; k++){
    W_pow *= W;
    rho_hat_pow *= rho_hat;
    log_det += rho_hat_pow * trace(W_pow) / k;
  }
  return (2. / m) * log_det + log(dot(IX_A_y, IX_A_y));
}
