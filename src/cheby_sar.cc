#include "cheby_sar.h"

using namespace arma;

ChebyshevSAR::ChebyshevSAR(const colvec &y, const mat &X, const sp_mat &W) :
  SAR(y, X, W) {
    cheby_poly_coeffs << 
      1 << 0 << 0 << endr << 
      0 << 1 << 0 << endr << 
      -1 << 0 << 2 << endr;
}

double
ChebyshevSAR::calc_log_det(double rho_hat){
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
  return dot(cpos * cheby_poly_coeffs, tdvec);
}
