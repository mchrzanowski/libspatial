#include <armadillo>
#include <iostream>
#include "sar.h"
#include "exact_sar.h"

using namespace arma;

int
main(int argc, char **argv){
  uword m = 100, n = 100;
  colvec y = randu<colvec>(m);
  mat W = randu<mat>(m, m);
  W += W.t();
  W.transform([] (double val) { return val > 1.75 ? 1 : 0; });
  mat X = randu<mat>(m, n);
  SAR *sar = new ExactSAR(W, X, y);
  sar->solve();

  double rho = sar->get_rho();
  const colvec beta = sar->get_beta();

  std::cout << "Rho: " << rho << std::endl;
  std::cout << "Sigma Squared: " << sar->get_sigma_sq() << std::endl;
  std::cout << "|beta|: " << norm(beta) << std::endl;

  colvec y_hat = solve(speye<sp_mat>(W.n_rows, W.n_cols) - rho * W, X * beta);
  std::cout << "RMSE: " << sqrt(sum(y_hat - y) / y.n_rows) << std::endl;

  delete sar;
  return 0;
}
