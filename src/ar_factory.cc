#include "ar_factory.h"
#include "cheby_car.h"
#include "cheby_sar.h"
#include "exact_car.h"
#include "exact_sar.h"

using namespace arma;

CAR *
get_CAR_solver(const colvec &y, const mat &X, const sp_mat &W, bool exact){
  if (exact){
    return new ExactCAR(y, X, W);
  }
  else {
    return new ChebyshevCAR(y, X, W);
  }
}

SAR *
get_SAR_solver(const colvec &y, const mat &X, const sp_mat &W, bool exact){
  if (exact){
    return new ExactSAR(y, X, W);
  }
  else {
    return new ChebyshevSAR(y, X, W);
  }
}
