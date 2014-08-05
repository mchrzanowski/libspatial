#include "exact_sar.h"
#include <assert.h>

using namespace arma;

ExactSAR::ExactSAR(const mat &W, const mat &X, const colvec &y) :
  SAR(y, X.n_rows, X.n_cols), W(W), X(X),
  IX(speye<sp_mat>(X.n_rows, X.n_cols) - X * inv(X.t() * X) * X.t()) {
    eigs = eig_sym(W);
}

void
ExactSAR::solve(){
  rho = golden_section_search(-1 + datum::eps, 0, 1 - datum::eps);
  const mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho * W;
  const colvec Ay = A * y;
  beta = arma::solve(X.t() * X, X.t() * Ay);
  const colvec val = IX * Ay;
  sigma_sq = dot(val, val) / m;
}

double
ExactSAR::log_likelihood(double rho_hat){
  const mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho_hat * W;
  const colvec IXAy = IX * A * y;
  // log_det(A) = \sum_{i=1}^n \log(1 - \rho\lambda_i)
  const double log_det = sum(log(1 - rho_hat * eigs));
  
  return (-2. / m) * log_det + log(dot(IXAy, IXAy));
}

/* from: http://en.wikipedia.org/wiki/Golden_section_search */
double
ExactSAR::golden_section_search(double a, double b, double c, double tol) {
  static const double phi = (1 + sqrt(5)) / 2;
  static const double resphi = 2 - phi;
  double x;
  if (c - b > b - a){
    x = b + resphi * (c - b);
  }
  else {
    x = b - resphi * (b - a);
  }
  if (abs(c - a) <= tol * (abs(b) + abs(x))){
    return (c + a) / 2; 
  }

  double ll_x = log_likelihood(x);
  double ll_b = log_likelihood(b);
  assert(ll_x != ll_b);
  if (ll_x < ll_b) {
    if (c - b > b - a) {
      return golden_section_search(b, x, c, tol);
    }
    else {
      return golden_section_search(a, x, b, tol);
    }
  }
  else {
    if (c - b > b - a) {
      return golden_section_search(a, b, x, tol);
    }
    else {
      return golden_section_search(x, b, c, tol);
    }
  }
}
