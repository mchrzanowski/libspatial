#include "exact_car.h"

using namespace arma;

ExactCAR::ExactCAR(const colvec &y, const mat &X, const sp_mat &W) :
  CAR(y, X, W), eigs(eig_sym(mat(W))) {}

double
ExactCAR::calc_log_det(double rho_hat){
  // log_det(A) = \sum_{i=1}^n \log(1 - \rho\lambda_i)
  return sum(log(1 - rho_hat * eigs));
}

