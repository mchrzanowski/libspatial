#include "exact_car.h"

using namespace arma;

ExactCAR::ExactCAR(const colvec &y, const mat &X, const sp_mat &W) :
  CAR(y, X, W), eigs(eig_sym(mat(W))) {}

double
ExactCAR::calc_log_det(double rho_hat){
  // log_det(A) = \sum_{i=1}^n \log(1 - \rho\lambda_i)
  return sum(log(1 - rho_hat * eigs));
}

double
ExactCAR::log_likelihood(){
  const sp_mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho * W;
  const colvec y_XB = y - X * beta;
  return sum(log(1 - rho * eigs)) / 2 - dot(y_XB, A * y_XB) / (2 * sigma_sq)
    - m * (log(2 * datum::pi) + log(sigma_sq)) / 2;
}
