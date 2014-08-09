#include <armadillo>
#include <iostream>
#include <string>
#include "exact_sar.h"
#include "taylor_sar.h"
#include "cheby_sar.h"
#include "CImg.h"
#include "create_w.h"
#include "car.h"
#include "exact_car.h"
#include "ar_model.h"

using namespace arma;

void
run_test(ARModel *sar, const std::string test_name){
  sar->solve();
  std::cout << test_name << std::endl;
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

  sp_mat W;
  create_4_neighbor_W(m, img.height(), W);  
  mat X = ones<mat>(m, n);

  wall_clock timer;

  timer.tic();
  SAR *sar = new ExactSAR(y, X, W);
  run_test(sar, std::string("Exact SAR: "));
  std::cout << "Runtime: " << timer.toc() << " secs. " << std::endl;
  std::cout << "Log-Likelihood: " << sar->log_likelihood() << std::endl;
  delete sar;

  std::cout << std::endl;
  
  timer.tic();
  sar = new ChebyshevSAR(y, X, W);
  run_test(sar, std::string("Chebyshev SAR: "));
  std::cout << "Runtime: " << timer.toc() << " secs. " << std::endl;
  std::cout << "Log-Likelihood: " << sar->log_likelihood() << std::endl;
  delete sar;

  std::cout << std::endl;
  
  CAR *car = new ExactCAR(y, X, W);
  run_test(car, std::string("Exact CAR: "));
  std::cout << "Runtime: " << timer.toc() << " secs. " << std::endl;
  std::cout << "Log-Likelihood: " << car->log_likelihood() << std::endl;
  delete car;

  return 0;
}
