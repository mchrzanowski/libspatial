#include "cheby_sar.h"

using namespace arma;

ChebyshevSAR::ChebyshevSAR(const sp_mat &W, const mat &X, const colvec &y) :
  SAR(y, X, W) {
    cheby_poly_coeffs << 
      1 << 0 << 0 << endr << 
      0 << 1 << 0 << endr << 
      -1 << 0 << 2 << endr;
}

double
ChebyshevSAR::rho_ll(double rho_hat){
  static const double td1 = 0;
  static const double td2 = accu(W % W);
  static const double npos = 3;
  static const colvec seq_1n_poss = {1, 2, 3};
  static const colvec x = cos(datum::pi * (seq_1n_poss - 0.5) / npos);

  rowvec cpos = zeros<rowvec>(npos);
  for (uword j = 0; j < npos; j++){
    cpos[j] = (2/npos) * sum(log(1 - rho_hat * x) % 
      cos(datum::pi * j * (seq_1n_poss - 0.5) / npos));
  }
  colvec tdvec;
  tdvec << m << endr << td1 << endr << td2 - 0.5 * m << endr;
  rowvec combo_term = cpos * cheby_poly_coeffs;
  return dot(combo_term, tdvec);
}

double
ChebyshevSAR::log_likelihood() {
  const sp_mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho * W;
  const colvec A_y = A * y;
  colvec val = A_y - X * beta;
  static const colvec eigs = eig_sym(mat(W));
  return sum(log(1 - rho * eigs)) - dot(val, val) / (2 * sigma_sq)
          - m * log(2 * datum::pi) / 2 - m * log(sigma_sq) / 2;
}
