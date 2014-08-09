#include "sar.h"
#include <cmath>

using namespace arma;

SAR::SAR(const colvec &y, const mat &X, const sp_mat &W) :
  ARModel(y, X, W),
  IX(speye<sp_mat>(X.n_rows, X.n_rows) - X * pinv(X.t() * X) * X.t()) {}

void 
SAR::solve(){
  calculate_rho();
  const sp_mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho * W;
  const colvec A_y = A * y;
  if (! arma::solve(beta, X.t() * X, X.t() * A_y) || ! is_finite(beta)){
    beta = zeros<colvec>(X.n_cols);  
  }
  const colvec IX_A_y = IX * A_y;
  sigma_sq = dot(IX_A_y, IX_A_y) / m;
}
