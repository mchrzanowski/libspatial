#include <armadillo>
#include <iostream>
#include "exact_sar.h"
#include "taylor_sar.h"
#include "cheby_sar.h"
#include "CImg.h"

using namespace arma;

void
run_test(SAR *sar){
  sar->solve();
  std::cout << "Rho: " << sar->get_rho() << std::endl;
  std::cout << "Sigma Squared: " << sar->get_sigma_sq() << std::endl;
  std::cout << "beta: " << sar->get_beta();
}

int
main(int argc, char **argv){
  arma_rng::set_seed_random();
  
  const cimg_library::CImg<unsigned char> img("../etc/mystery.png");
  colvec y = zeros<colvec>(img.size());
  uword m = y.n_rows, n = 1;

  for (int i = 0; i < img.width(); i++){
    for (int j = 0; j < img.height(); j++){
      y[i * img.height() + j] = img(i, j);
    }
  }
  y /= 255;

  //std::cout << mean(y) << std::endl;

  sp_mat W = zeros<sp_mat>(m, m);
  colvec d = zeros<colvec>(m);
  int img_width = img.width();
  for (int i = 0; i < img_width; i++){
    for (int j = 0; j < img_width; j++){
      uword sourceIndx = j * img_width + i, destIndx;
      if (j + 1 < img_width){
        destIndx = (j + 1) * img_width + i;
        W(sourceIndx, destIndx) = 1;
        d[sourceIndx]++;
      } 
      if (i + 1 < img_width) {
        destIndx = j * img_width + i + 1;
        W(sourceIndx, destIndx) = 1;
        d[sourceIndx]++;
      }
      if (i - 1 >= 0) {
        destIndx = j * img_width + i - 1;
        W(sourceIndx, destIndx) = 1;
        d[sourceIndx]++;
      }
      if (j - 1 >= 0){
        destIndx = (j - 1) * img_width + i;
        W(sourceIndx, destIndx) = 1;
        d[sourceIndx]++;
      }
    }
  }
  const mat D = diagmat(1 / d);
  W = D * W;

  mat X = ones<mat>(m, n);

  wall_clock timer;

  /*timer.tic();
  SAR *sar = new ExactSAR(W, X, y);
  std::cout << "Exact SAR: " << std::endl;
  run_test(sar);
  std::cout << "Runtime: " << timer.toc() << " secs. " << std::endl;
  std::cout << "Log-Likelihood: " << sar->log_likelihood() << std::endl;
  delete sar;*/

  timer.tic();
  SAR *sar = new ChebyshevSAR(W, X, y);
  std::cout << "Chebyshev SAR: " << std::endl;
  run_test(sar);
  std::cout << "Runtime: " << timer.toc() << " secs. " << std::endl;
  std::cout << "Log-Likelihood: " << sar->log_likelihood() << std::endl;
  delete sar;

  return 0;
}
