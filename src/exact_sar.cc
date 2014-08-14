#include "exact_sar.h"

using namespace arma;

ExactSAR::ExactSAR(const colvec &y, const mat &X, const sp_mat &W) :
  SAR(y, X, W), eigs(eig_sym(mat(W))) {}

double
ExactSAR::calc_log_det(double rho_hat){
  // log_det(A) = \sum_{i=1}^n \log(1 - \rho\lambda_i)
  return sum(log(1 - rho_hat * eigs));
}
