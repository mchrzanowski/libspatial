#include <armadillo>
#include <iostream>
#include "exact_sar.h"
#include "taylor_sar.h"
#include "CImg.h"

using namespace arma;
using namespace cimg_library;

int
main(int argc, char **argv){
  arma_rng::set_seed_random();
  
  const CImg<unsigned char> img("../etc/mystery.png");
  colvec y = zeros<colvec>(img.size());
  uword m = y.n_rows, n = 1;

  for (int i = 0; i < img.width(); i++){
    for (int j = 0; j < img.height(); j++){
      y[i * img.height() + j] = img(i, j);
    }
  }
  y /= 255;

  std::cout << mean(y) << std::endl;

  mat W = zeros<mat>(m, m);
  int img_width = img.width();
  for (int i = 0; i < img_width; i++){
    for (int j = 0; j < img_width; j++){
      uword sourceIndx = j * img_width + i, destIndx;
      if (j + 1 < img_width){
        destIndx = (j + 1) * img_width + i;
        W(sourceIndx, destIndx) = 1;
      } 
      if (i + 1 < img_width) {
        destIndx = j * img_width + i + 1;
        W(sourceIndx, destIndx) = 1;
      }
      if (i - 1 >= 0) {
        destIndx = j * img_width + i - 1;
        W(sourceIndx, destIndx) = 1;
      }
      if (j - 1 >= 0){
        destIndx = (j - 1) * img_width + i;
        W(sourceIndx, destIndx) = 1;
      }
    }
  }
  W.each_col() /= sum(W, 1);

  mat X = zeros<mat>(m, n);

  SAR *sar = new ExactSAR(W, X, y);
  sar->solve();
  std::cout << "Exact SAR: " << std::endl;
  std::cout << "Rho: " << sar->get_rho() << std::endl;
  std::cout << "Sigma Squared: " << sar->get_sigma_sq() << std::endl;
  std::cout << "intercept: " << sar->get_mu() << std::endl;
  std::cout << "beta: " << sar->get_beta() << std::endl;
  delete sar;

  /*sar = new TaylorSAR(W, X, y);
  sar->solve();
  std::cout << "Taylor SAR Approx: " << std::endl;
  std::cout << "Rho: " << sar->get_rho() << std::endl;
  std::cout << "Sigma Squared: " << sar->get_sigma_sq() << std::endl;
  std::cout << "|beta|: " << sar->get_beta() << std::endl;
  delete sar;*/

  return 0;
}
