#include "taylor_sar.h"

using namespace arma;

TaylorSAR::TaylorSAR(const sp_mat &W, const mat &X, const colvec &y) :
  SAR(y, X, W) {}

double
TaylorSAR::calc_log_det(double rho_hat){
  double log_det = 0;
  sp_mat W_pow = speye<sp_mat>(W.n_rows, W.n_cols);
  double rho_hat_pow = 1;
  for (unsigned k = 1; k <= 30; k++){
    W_pow *= W;
    rho_hat_pow *= rho_hat;
    log_det += rho_hat_pow * trace(W_pow) / k;
  }
  return log_det;
}
