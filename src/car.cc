#include "car.h"
#include <cmath>

using namespace arma;

CAR::CAR(const colvec &y, const mat &X, const sp_mat &W) : ARModel(y, X, W) {}

void
CAR::solve(){
  calculate_rho();
  const sp_mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho * W;
  if (! arma::solve(beta, X.t() * A * X, X.t() * y) || ! is_finite(beta)){
    beta = zeros<colvec>(X.n_cols);  
  }
  const colvec y_XB = y - X * beta;
  sigma_sq = dot(y_XB, A * y_XB) / m;
}
