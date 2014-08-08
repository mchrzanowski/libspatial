#include "sar.h"

class TaylorSAR : public SAR {
public:
  TaylorSAR(const arma::sp_mat &W, const arma::mat &X, const arma::colvec &y);
  double log_likelihood();

protected:
  double rho_ll(double rho_hat);
};
