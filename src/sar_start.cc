#include <armadillo>
#include <iostream>
#include "sar.h"
#include "exact_sar.h"

using namespace arma;

int
main(int argc, char **argv){
  arma_rng::set_seed_random();
  uword m = 100, n = 100;
  mat W = randu<mat>(m, m);
  W += W.t();
  W.transform([] (double val) { return val > 1.75 ? 1 : 0; });
  mat X = randu<mat>(m, n);
  colvec y = 10 * randn<colvec>(m);
  SAR *sar = new ExactSAR(W, X, y);
  sar->solve();

  std::cout << "Rho: " << sar->get_rho() << std::endl;
  std::cout << "Sigma Squared: " << sar->get_sigma_sq() << std::endl;
  std::cout << "|beta|: " << norm(sar->get_beta()) << std::endl;

  delete sar;
  return 0;
}
