#include <armadillo>
#include <iostream>
#include <string>
#include "CImg.h"
#include "ar_factory.h"
#include "create_w.h"

using namespace arma;

void
run_test(ARModel *ar, wall_clock timer, const std::string test_name){
  ar->solve();
  std::cout << test_name << std::endl;
  std::cout << "Rho: " << ar->get_rho() << std::endl;
  std::cout << "Sigma Squared: " << ar->get_sigma_sq() << std::endl;
  std::cout << "beta: " << ar->get_beta();
  std::cout << "Runtime: " << timer.toc() << " secs. " << std::endl;
  std::cout << "Log-Likelihood: " << ar->log_likelihood() << std::endl;
  std::cout << std::endl;
}

int
main(int argc, char **argv){
  arma_rng::set_seed(0);
  const cimg_library::CImg<unsigned char> img("../tests/mystery.png");
  colvec y = zeros<colvec>(img.size());
  const uword m = y.n_rows, n = 1;

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
  ARModel *ar = get_SAR_solver(y, X, W, true);
  run_test(ar, timer, std::string("Exact SAR: "));
  delete ar;

  timer.tic();
  ar = get_SAR_solver(y, X, W);
  run_test(ar, timer, std::string("Chebyshev SAR: "));
  delete ar;

  timer.tic();
  ar = get_CAR_solver(y, X, W, true);
  run_test(ar, timer, std::string("Exact CAR: "));
  delete ar;

  timer.tic();
  ar = get_CAR_solver(y, X, W);
  run_test(ar, timer, std::string("Chebyshev CAR: "));
  delete ar;

  return 0;
}
