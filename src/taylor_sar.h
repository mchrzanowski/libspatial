#include "sar.h"

class TaylorSAR : public SAR {
public:
  TaylorSAR(const arma::sp_mat &W, const arma::mat &X, const arma::colvec &y);

protected:
  double calc_log_det(double rho_hat);
};
