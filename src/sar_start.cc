#include <armadillo>
#include <iostream>
#include "sar.h"

using namespace arma;

int
main(int argc, char **argv){
  std::cout << "HI" << std::endl;

  uword m = 100, n = 100;
  colvec y = randu<colvec>(m);
  mat W = randu<mat>(m, m);
  W += W.t();
  W.transform([] (double val) { return val > 1.5 ? 1 : 0; });
  mat X = randu<mat>(m, n);
  ExactSAR *sar = new ExactSAR(W, X, y);
  sar->solve();
  delete sar;

  return 0;
}
